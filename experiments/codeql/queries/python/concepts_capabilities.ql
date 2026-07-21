/**
 * @name Skill capability events (Tier 1 — Concepts)
 * @description Detect capability-relevant behaviors using CodeQL's Concepts
 *              abstraction layer. Each Concept class covers all libraries
 *              modeled by CodeQL (e.g., Http::Client::Request covers
 *              requests, httpx, urllib, aiohttp, pycurl, etc.).
 *              SkillRecon should never hardcode API names for behaviors
 *              that Concepts already abstract.
 * @kind problem
 * @problem.severity recommendation
 * @id skillrecon/python/concepts-capabilities
 */

import python
import semmle.python.Concepts
import semmle.python.dataflow.new.DataFlow

// ---------------------------------------------------------------------------
// Tier 1: Concepts-based capability detection
// ---------------------------------------------------------------------------

predicate isConceptCapability(DataFlow::Node node, string capType, string detail) {
  // --- Network ---
  // Http::Client::Request covers: requests, httpx, urllib, aiohttp, pycurl,
  // libtaxii, urllib2, urllib3, and stdlib http.client
  exists(Http::Client::Request req |
    node = req and
    capType = "http_request" and
    detail = "HTTP client request"
  )
  or
  // --- Process execution ---
  // SystemCommandExecution covers: os.system, os.popen, subprocess.*,
  // pexpect, paramiko exec_command, fabric, invoke
  exists(SystemCommandExecution cmd |
    node = cmd and
    capType = "shell_exec" and
    detail = "system command execution"
  )
  or
  // --- File system ---
  // FileSystemAccess covers: open, pathlib, anyio, aiofiles, aiofile,
  // lxml, werkzeug, flask, fastapi, sanic, starlette, cherrypy, aiohttp, baize
  (
    exists(FileSystemWriteAccess fsw |
      node = fsw and
      capType = "file_write" and
      detail = "file system write"
    )
    or
    exists(FileSystemAccess fsa |
      not fsa instanceof FileSystemWriteAccess and
      node = fsa and
      capType = "file_read" and
      detail = "file system read"
    )
  )
  or
  // --- Code execution ---
  // CodeExecution covers: eval, exec, compile, pandas.eval, pandas.query
  exists(CodeExecution ce |
    node = ce and
    capType = "eval_exec" and
    detail = "code execution"
  )
  or
  // --- SQL ---
  // SqlExecution covers: PEP249 (sqlite3, psycopg2, mysql-connector, etc.),
  // SQLAlchemy, Django ORM, asyncpg, aiomysql, peewee, pandas.read_sql
  exists(SqlExecution sql |
    node = sql and
    capType = "sql_exec" and
    detail = "SQL execution"
  )
  or
  // --- Deserialization ---
  // Decoding covers: pickle, yaml, json, base64, torch, numpy, ujson,
  // joblib, ruamel.yaml, simplejson, jsonpickle, dill, pymongo BSON
  // Only flag unsafe formats (pickle, yaml, marshal) — not json/base64.
  exists(Decoding dec |
    dec.getFormat().regexpMatch("(?i)pickle|yaml|marshal") and
    node = dec and
    capType = "deserialization" and
    detail = "unsafe deserialization (" + dec.getFormat() + ")"
  )
  or
  // --- Encoding / data send ---
  exists(Encoding enc |
    node = enc and
    capType = "data_encode_send" and
    detail = "data encoding (" + enc.getFormat() + ")"
  )
}

from DataFlow::Node node, string capType, string detail
where isConceptCapability(node, capType, detail)
select node,
  "capType=" + capType + " | detail=" + detail + " | tier=concepts"
