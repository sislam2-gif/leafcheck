// Service worker for LeafCheck Expert. Makes the app installable and opens instantly.
// The page itself is fetched fresh when online (so updates show up right away) and
// falls back to the saved copy when Atlas can't be reached. API calls always go to Atlas.
const CACHE = "leafcheck-expert-v1";
const SHELL = ["/", "/static/manifest.json", "/static/icons/icon-192.png", "/static/icons/icon-512.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.pathname.startsWith("/api/")) return;   // never cache API answers
  e.respondWith(
    fetch(e.request).then(res => {
      if (res.ok && SHELL.includes(url.pathname)) caches.open(CACHE).then(c => c.put(e.request, res.clone()));
      return res;
    }).catch(() => caches.match(e.request).then(hit => hit || caches.match("/")))
  );
});
