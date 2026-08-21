/**
 * RecSys Cloudflare Worker.
 *
 * Responsibilities:
 *  - Serve the static UI (single-page `public/index.html`) via the ASSETS binding.
 *  - Proxy every API route to the Render backend (API_RENDER_URL), forwarding the
 *    path, query string, method and body unchanged. The UI uses same-origin relative
 *    paths (API_BASE = ''), so /search, /recommend, /chat/recommend, /event, /health
 *    and /metrics must be handled on this same origin.
 *
 * Env bindings (see wrangler.jsonc):
 *  - API_RENDER_URL   : the Render web service base URL (no trailing slash)
 *  - ASSETS           : static asset binding pointing at ./public
 */

export interface Env {
	API_RENDER_URL: string;
	ASSETS: Fetcher;
}

const API_PREFIXES = [
	'/api',
	'/health',
	'/metrics',
	'/search',
	'/recommend',
	'/chat/recommend',
	'/popular',
	'/event',
];

function isApiPath(pathname: string): boolean {
	if (pathname === '/') {
		return false;
	}
	return API_PREFIXES.some((p) => pathname === p || pathname.startsWith(`${p}/`));
}

/**
 * Forward a request to the backend, preserving method/headers/body and relaxing CORS.
 */
async function proxy(request: Request, target: URL): Promise<Response> {
	const headers = new Headers(request.headers);
	headers.delete('host');
	headers.set('accept', request.headers.get('accept') ?? '*/*');

	const init: RequestInit = {
		method: request.method,
		headers,
		redirect: 'follow',
	};
	if (request.method !== 'GET' && request.method !== 'HEAD') {
		init.body = request.body;
	}

	try {
		const upstream = await fetch(target.toString(), init);
		const respHeaders = new Headers(upstream.headers);
		respHeaders.set('access-control-allow-origin', '*');
		return new Response(upstream.body, {
			status: upstream.status,
			statusText: upstream.statusText,
			headers: respHeaders,
		});
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);
		return new Response(JSON.stringify({ error: 'upstream unreachable', detail: msg }), {
			status: 502,
			headers: { 'content-type': 'application/json', 'access-control-allow-origin': '*' },
		});
	}
}

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const url = new URL(request.url);

		if (isApiPath(url.pathname)) {
			// Rewrite /api/foo -> /foo on the backend (strip the /api prefix), but keep
			// the other API routes verbatim since the UI calls them as bare paths.
			const path = url.pathname.startsWith('/api')
				? url.pathname.slice('/api'.length) || '/'
				: url.pathname;
			const target = new URL(path + url.search, env.API_RENDER_URL);
			return proxy(request, target);
		}

		// Static UI (public/index.html served at "/" and other non-API paths).
		return env.ASSETS.fetch(request);
	},
} satisfies ExportedHandler<Env>;
