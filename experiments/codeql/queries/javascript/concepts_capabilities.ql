/**
 * @name Skill capability events (Tier 1 — Concepts)
 * @description Detect capability-relevant behaviors in JavaScript/TypeScript
 *              using CodeQL's Concepts and framework models.
 *              Note: JS Concepts are distributed across frameworks/*.qll,
 *              unlike Python's centralized Concepts.qll.
 * @kind problem
 * @problem.severity recommendation
 * @id skillrecon/javascript/concepts-capabilities
 */

import javascript
import semmle.javascript.frameworks.WebSocket

// ---------------------------------------------------------------------------
// Tier 1: Concepts-based capability detection for JS/TS
// ---------------------------------------------------------------------------

predicate isConceptCapability(DataFlow::Node node, string capType, string detail) {
  // --- Network ---
  exists(ClientWebSocket::ClientSocket ws |
    node = ws and
    capType = "websocket" and
    detail = "websocket connection"
  )
  or
  // ClientRequest covers: fetch, axios, got, superagent, request,
  // node-fetch, XMLHttpRequest, needle, apollo-client, etc. (15+ libs)
  exists(ClientRequest req |
    node = req and
    not req instanceof ClientWebSocket::ClientSocket and
    capType = "http_request" and
    detail = "HTTP client request"
  )
  or
  // --- Process execution ---
  // SystemCommandExecution covers: child_process.exec/spawn/fork,
  // execa, shelljs, cross-spawn, ssh2, etc. (10+ libs)
  exists(SystemCommandExecution cmd |
    node = cmd and
    capType = "shell_exec" and
    detail = "system command execution"
  )
  or
  // --- File system ---
  // FileSystemAccess / FileSystemReadAccess / FileSystemWriteAccess
  // covers: fs.*, path operations
  (
    exists(FileSystemWriteAccess fsw |
      node = fsw and
      capType = "file_write" and
      detail = "file system write"
    )
    or
    exists(FileSystemReadAccess fsr |
      node = fsr and
      capType = "file_read" and
      detail = "file system read"
    )
  )
  or
  // --- Database ---
  // DatabaseAccess covers: mysql, pg, sqlite3, knex, sequelize,
  // mssql, better-sqlite3, etc. (10+ libs)
  exists(DatabaseAccess db |
    node = db and
    capType = "sql_exec" and
    detail = "database access"
  )
  or
  // --- Crypto ---
  exists(Cryptography::CryptographicOperation crypto |
    node = crypto and
    capType = "data_encode_send" and
    detail = "cryptographic operation"
  )
}

from DataFlow::Node node, string capType, string detail
where isConceptCapability(node, capType, detail)
select node,
  "capType=" + capType + " | detail=" + detail + " | tier=concepts"
