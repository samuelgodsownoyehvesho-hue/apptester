/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The benchmark drives this app through a browser and an HTTP client, so it
  // must not depend on a production build being present.
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
