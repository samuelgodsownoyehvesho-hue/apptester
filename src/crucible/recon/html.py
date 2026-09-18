"""HTML analysis for reconnaissance.

Uses the standard library parser rather than a third-party one. The output we
need is narrow — links, forms, controls, headings, and whether each control has
an accessible name — and a full DOM library would be a large dependency for
that.

The accessible-name analysis is deliberately part of recon rather than the
accessibility oracle. Discovering that a control is unlabelled is a *fact*;
deciding that it is a defect is a judgement. Keeping facts and judgements in
separate layers is what stops the oracle from depending on how it happened to
discover something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

#: Elements that can carry an accessible name in their own right.
_NAME_ATTRS = ("aria-label", "aria-labelledby", "title")

#: Attributes that never provide an accessible name. A placeholder is a hint,
#: not a label, and counting it would mask exactly the defect we want to find.
_NOT_A_NAME = ("placeholder",)

#: How many interactive elements a single page contributes to the app map. Set
#: high on purpose: the interaction lane exists to exercise everything a page
#: offers, and a low ceiling would silently reduce coverage to "the first few
#: controls", which reads as a clean page rather than a truncated one.
MAX_ELEMENTS_PER_PAGE = 400

#: Input types with nothing for a person (or a bot) to interact with.
_NON_INTERACTIVE_INPUTS = frozenset({"hidden"})


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """A single form control."""

    kind: str
    name: str | None = None
    field_id: str | None = None
    input_type: str | None = None
    placeholder: str | None = None
    required: bool = False
    #: True when the control has a usable accessible name. Placeholders and
    #: `<p>` text before the input do not count.
    has_label: bool = False
    label_text: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "id": self.field_id,
            "type": self.input_type,
            "required": self.required,
            "has_label": self.has_label,
            "label_text": self.label_text,
        }


@dataclass(frozen=True, slots=True)
class ElementSpec:
    """One interactive element discovered on a page.

    ``selector`` is best-effort: a static parse cannot know what the rendered
    DOM will look like, so the interaction lane treats it as a hint and falls
    back to matching on the label when it misses.
    """

    #: ``button``, ``link``, ``input``, ``select``, ``textarea`` or
    #: ``clickable`` for a non-semantic element with a click handler.
    kind: str
    selector: str
    label: str = ""
    element_type: str | None = None
    in_form: bool = False
    href: str | None = None
    options: tuple[str, ...] = ()
    has_click_handler: bool = False
    #: How it was found: ``html`` for the static parse, ``browser`` for the
    #: live-DOM inventory that sees JavaScript-rendered controls.
    source: str = "html"

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "selector": self.selector,
            "label": self.label,
            "type": self.element_type,
            "in_form": self.in_form,
            "href": self.href,
            "options": list(self.options),
            "has_click_handler": self.has_click_handler,
            "source": self.source,
        }


@dataclass(slots=True)
class FormSpec:
    """A form and its controls."""

    action: str
    method: str = "get"
    fields: list[FieldSpec] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "method": self.method,
            "fields": [f.as_dict() for f in self.fields],
        }


@dataclass(slots=True)
class PageFacts:
    """Structured observations about one page."""

    url: str
    title: str | None = None
    #: Same-origin, normalised link targets.
    links: list[str] = field(default_factory=list)
    #: Link targets on other origins, recorded but never crawled.
    external_links: list[str] = field(default_factory=list)
    forms: list[FormSpec] = field(default_factory=list)
    #: Controls not inside any form, which is common in modern UIs.
    loose_fields: list[FieldSpec] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    buttons: list[str] = field(default_factory=list)
    #: Every interactive element on the page, in document order. This is what
    #: makes a generic "press everything" lane possible: without it the browser
    #: lane has nothing to click.
    elements: list[ElementSpec] = field(default_factory=list)
    text_excerpt: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def unlabelled_controls(self) -> list[FieldSpec]:
        """Controls with no accessible name, across forms and loose fields."""
        everything = [f for form in self.forms for f in form.fields] + self.loose_fields
        # Buttons and hidden inputs are excluded: a submit button's value is
        # usually its name, and hidden fields are not user-facing.
        return [
            field
            for field in everything
            if not field.has_label
            and field.kind not in {"hidden", "submit", "button", "image", "reset"}
        ]

    def as_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "title": self.title,
            "links": self.links,
            "external_links": self.external_links,
            "forms": [f.as_dict() for f in self.forms],
            "loose_fields": [f.as_dict() for f in self.loose_fields],
            "headings": self.headings,
            "buttons": self.buttons,
            "elements": [element.as_dict() for element in self.elements],
            "unlabelled_control_count": len(self.unlabelled_controls),
        }


def _selector_for(tag: str, attrs: dict[str, str | None]) -> str:
    """Best-effort CSS selector for a control, from stable attributes only.

    Position-based selectors are avoided: they break the moment a page renders
    differently from its source, and a selector that silently matches the wrong
    element is worse than none at all. When nothing stable exists the label is
    the fallback, resolved by the interaction lane against the live DOM.
    """
    for attribute in ("id", "name", "aria-label"):
        value = attrs.get(attribute)
        if value and '"' not in value and "'" not in value and "\\" not in value:
            if attribute == "id":
                return f"#{value}"
            return f'{tag}[{attribute}="{value}"]'
    return ""


class _PageParser(HTMLParser):
    """Single-pass parser accumulating the facts we care about."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.facts = PageFacts(url=base_url)

        self._label_for_ids: set[str] = set()
        self._label_text_by_id: dict[str, str] = {}
        self._label_depth = 0
        self._current_label_for: str | None = None
        self._label_buffer: list[str] = []

        # Parallel drafts of every field, so labels that appear *after* the
        # control can still be resolved once parsing finishes.
        self._field_drafts: list[dict[str, object]] = []

        self._in_title = False
        self._title_buffer: list[str] = []
        self._heading_tag: str | None = None
        self._heading_buffer: list[str] = []
        self._text_buffer: list[str] = []
        self._form_depth = 0
        #: Interactive controls, collected as drafts because a control's visible
        #: text only arrives after its start tag.
        self._element_drafts: list[dict[str, object]] = []
        self._open_element: dict[str, object] | None = None
        self._open_select: dict[str, object] | None = None
        self._option_buffer: list[str] = []

    def _start_element(
        self,
        tag: str,
        kind: str,
        attrs: dict[str, str | None],
        *,
        href: str | None = None,
    ) -> None:
        """Begin recording one interactive control."""
        if len(self._element_drafts) >= MAX_ELEMENTS_PER_PAGE:
            return
        draft: dict[str, object] = {
            "tag": tag,
            "kind": kind,
            "selector": _selector_for(tag, attrs),
            "label": (attrs.get("aria-label") or attrs.get("title") or "").strip(),
            "element_type": attrs.get("type"),
            "in_form": self._form_depth > 0,
            "href": href,
            "options": [],
            "has_click_handler": "onclick" in attrs,
            "text": [],
        }
        self._element_drafts.append(draft)
        self._open_element = draft

    # -- helpers ---------------------------------------------------------

    def _record_field(
        self,
        *,
        kind: str,
        attrs: dict[str, str | None],
        wrapped_in_label: bool,
    ) -> None:
        draft: dict[str, object] = {
            "kind": kind,
            "name": attrs.get("name"),
            "field_id": attrs.get("id"),
            "input_type": attrs.get("type"),
            "placeholder": attrs.get("placeholder"),
            "required": "required" in attrs,
            "aria": [attrs.get(a) for a in _NAME_ATTRS if attrs.get(a)],
            "wrapped": wrapped_in_label,
            "label_text_hint": (attrs.get("aria-label") or None),
        }

        target = (
            self.facts.forms[-1].fields if self.facts.forms and self._form_depth > 0
            else self.facts.loose_fields
        )
        index = len(self._field_drafts)
        self._field_drafts.append(draft)

        spec = FieldSpec(
            kind=kind,
            name=draft["name"] if isinstance(draft["name"], str) else None,
            field_id=draft["field_id"] if isinstance(draft["field_id"], str) else None,
            input_type=draft["input_type"] if isinstance(draft["input_type"], str) else None,
            placeholder=draft["placeholder"] if isinstance(draft["placeholder"], str) else None,
            required=bool(draft["required"]),
        )
        target.append(spec)
        draft["_spec"] = spec
        draft["_index"] = index

    # -- parser hooks ----------------------------------------------------

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs_list)

        if tag == "title":
            self._in_title = True
            return

        if tag in {"h1", "h2", "h3"}:
            self._heading_tag = tag
            self._heading_buffer = []
            return

        if tag == "label":
            self._label_depth += 1
            self._current_label_for = attrs.get("for")
            self._label_buffer = []
            return

        if tag == "a":
            href = attrs.get("href")
            if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                absolute = urljoin(self.base_url, href)
                if _same_origin(absolute, self.base_url):
                    normalized = _normalize(absolute)
                    if normalized not in self.facts.links:
                        self.facts.links.append(normalized)
                elif absolute not in self.facts.external_links:
                    self.facts.external_links.append(absolute)
                self._start_element("a", "link", attrs, href=href)
            return

        if tag == "form":
            self._form_depth += 1
            action_raw = attrs.get("action") or self.base_url
            self.facts.forms.append(
                FormSpec(
                    action=urljoin(self.base_url, action_raw),
                    method=(attrs.get("method") or "get").lower(),
                )
            )
            return

        if tag == "input":
            input_type = (attrs.get("type") or "text").lower()
            if input_type in _NON_INTERACTIVE_INPUTS:
                return
            self._record_field(kind=input_type, attrs=attrs, wrapped_in_label=self._label_depth > 0)
            self._start_element("input", "input", attrs)
            return

        if tag in {"select", "textarea"}:
            self._record_field(kind=tag, attrs=attrs, wrapped_in_label=self._label_depth > 0)
            self._start_element(tag, "select" if tag == "select" else "textarea", attrs)
            if tag == "select":
                self._open_select = self._element_drafts[-1]
            return

        if tag == "option" and self._open_select is not None:
            self._option_buffer = []
            return

        if tag == "button":
            label = (attrs.get("aria-label") or "").strip()
            if label:
                self.facts.buttons.append(label)
            self._start_element("button", "button", attrs)
            return

        # A non-semantic element wired up with a click handler is still a
        # control a visitor can use, and a very common way to build one.
        if "onclick" in attrs and tag not in {"html", "body", "form"}:
            self._start_element(tag, "clickable", attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            self.facts.title = "".join(self._title_buffer).strip() or None
            self._title_buffer = []
            return

        if tag in {"h1", "h2", "h3"} and self._heading_tag == tag:
            heading = " ".join("".join(self._heading_buffer).split())
            if heading:
                self.facts.headings.append(heading)
            self._heading_tag = None
            return

        if tag == "label":
            text = " ".join("".join(self._label_buffer).split())
            if self._current_label_for:
                self._label_for_ids.add(self._current_label_for)
                if text:
                    self._label_text_by_id[self._current_label_for] = text
            self._label_depth = max(0, self._label_depth - 1)
            self._current_label_for = None
            self._label_buffer = []
            return

        if tag == "form":
            self._form_depth = max(0, self._form_depth - 1)
            return

        if tag == "option" and self._open_select is not None:
            text = " ".join("".join(self._option_buffer).split())
            if text:
                options = self._open_select.get("options")
                if isinstance(options, list):
                    options.append(text)
            self._option_buffer = []
            return

        # Close out a control whose label is its own text: buttons and links
        # written as `<button>Save</button>` carry no attributes to match on.
        open_element = self._open_element
        if open_element is not None and open_element.get("tag") == tag:
            if not open_element.get("label"):
                raw_text = open_element.get("text")
                parts = raw_text if isinstance(raw_text, list) else []
                text = " ".join("".join(str(part) for part in parts).split())
                open_element["label"] = text[:80]
            self._open_element = None
            if tag == "select":
                self._open_select = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_buffer.append(data)
        if self._heading_tag is not None:
            self._heading_buffer.append(data)
        if self._label_depth > 0:
            self._label_buffer.append(data)
        if not data.isspace():
            self._text_buffer.append(data)
        open_element = self._open_element
        if open_element is not None:
            text = open_element.get("text")
            if isinstance(text, list):
                text.append(data)
        if self._open_select is not None:
            self._option_buffer.append(data)

    def finish(self) -> PageFacts:
        """Resolve labels now that the whole document has been seen."""
        for draft in self._field_drafts:
            spec = draft.get("_spec")
            if not isinstance(spec, FieldSpec):
                continue

            field_id = spec.field_id
            aria_present = bool(draft.get("aria"))
            wrapped = bool(draft.get("wrapped"))
            referenced = field_id is not None and field_id in self._label_for_ids

            # Bound to a local so the type narrows to str: reading it through
            # the draft mapping twice defeats narrowing and yields object.
            hint_raw = draft.get("label_text_hint")
            hint = hint_raw if isinstance(hint_raw, str) else None
            label_text = self._label_text_by_id.get(field_id or "") or hint

            # A FieldSpec is frozen, so rebuild it with the resolved verdict.
            resolved = FieldSpec(
                kind=spec.kind,
                name=spec.name,
                field_id=spec.field_id,
                input_type=spec.input_type,
                placeholder=spec.placeholder,
                required=spec.required,
                has_label=aria_present or wrapped or referenced,
                label_text=label_text,
            )

            for container in (*self.facts.forms,):
                for position, existing in enumerate(container.fields):
                    if existing is spec:
                        container.fields[position] = resolved
            for position, existing in enumerate(self.facts.loose_fields):
                if existing is spec:
                    self.facts.loose_fields[position] = resolved

        for draft in self._element_drafts:
            options = draft.get("options")
            option_texts = options if isinstance(options, list) else []
            element_type = draft.get("element_type")
            href = draft.get("href")
            self.facts.elements.append(
                ElementSpec(
                    kind=str(draft["kind"]),
                    selector=str(draft.get("selector") or ""),
                    label=str(draft.get("label") or ""),
                    element_type=element_type if isinstance(element_type, str) else None,
                    in_form=bool(draft.get("in_form")),
                    href=href if isinstance(href, str) else None,
                    options=tuple(str(option) for option in option_texts),
                    has_click_handler=bool(draft.get("has_click_handler")),
                    source="html",
                )
            )

        self.facts.text_excerpt = " ".join(" ".join(self._text_buffer).split())[:2000]
        return self.facts


def _same_origin(candidate: str, base: str) -> bool:
    try:
        a, b = urlsplit(candidate), urlsplit(base)
    except ValueError:
        return False
    return (a.scheme, a.netloc) == (b.scheme, b.netloc)


def _normalize(url: str) -> str:
    """Strip the fragment, which never changes what the server returns."""
    parts = urlsplit(url)
    return parts._replace(fragment="").geturl()


def extract_page_facts(html: str, base_url: str) -> PageFacts:
    """Parse ``html`` and return structured observations about the page."""
    parser = _PageParser(base_url)
    parser.feed(html)
    parser.close()
    return parser.finish()
