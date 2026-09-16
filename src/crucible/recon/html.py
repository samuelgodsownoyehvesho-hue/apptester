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
            "unlabelled_control_count": len(self.unlabelled_controls),
        }


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
            self._record_field(
                kind=(attrs.get("type") or "text").lower(),
                attrs=attrs,
                wrapped_in_label=self._label_depth > 0,
            )
            return

        if tag in {"select", "textarea"}:
            self._record_field(kind=tag, attrs=attrs, wrapped_in_label=self._label_depth > 0)
            return

        if tag == "button":
            label = (attrs.get("aria-label") or "").strip()
            if label:
                self.facts.buttons.append(label)

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

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_buffer.append(data)
        if self._heading_tag is not None:
            self._heading_buffer.append(data)
        if self._label_depth > 0:
            self._label_buffer.append(data)
        if not data.isspace():
            self._text_buffer.append(data)

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
