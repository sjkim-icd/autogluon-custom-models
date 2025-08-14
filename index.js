import http from "http";

console.log("[MCP] server booting...");

const server = http.createServer((_, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("MCP server is running\n");
});

server.listen(3000, () => {
  console.log("[MCP] listening on http://localhost:3000");
});
