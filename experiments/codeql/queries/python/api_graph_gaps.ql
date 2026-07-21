/**
 * @name Skill capability events (Tier 2 — API-Graph gaps)
 * @description Detect behaviors NOT covered by CodeQL Concepts but important
 *              for SkillRecon's capability taxonomy. Uses API-Graph to match
 *              specific APIs. These queries are gap-fillers — if CodeQL
 *              upstream promotes a behavior to a Concept, the corresponding
 *              predicate here should be retired.
 * @kind problem
 * @problem.severity recommendation
 * @id skillrecon/python/api-graph-gaps
 */

import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.ApiGraphs

// ---------------------------------------------------------------------------
// Tier 2: API-Graph gap fillers for Concepts blind spots
// ---------------------------------------------------------------------------

// --- Environment variable access ---
// Not a security sink in CodeQL's model, but critical for SkillRecon
// (credential leakage, configuration tampering)
predicate isEnvVarAccess(DataFlow::Node node, string detail) {
  // os.environ.get("KEY")
  exists(API::CallNode call |
    call =
      API::moduleImport("os").getMember("environ").getMember("get").getACall() and
    node = call and
    detail = "os.environ.get"
  )
  or
  // os.getenv("KEY")
  exists(API::CallNode call |
    call = API::moduleImport("os").getMember("getenv").getACall() and
    node = call and
    detail = "os.getenv"
  )
  or
  // os.environ["KEY"] (subscript access)
  exists(API::Node sub |
    sub = API::moduleImport("os").getMember("environ").getASubscript() and
    node = sub.asSink() and
    detail = "os.environ[]"
  )
}

// --- Dynamic import ---
// Not a typical security pattern, but relevant for supply-chain and
// code injection analysis in agent skills
predicate isDynamicImport(DataFlow::Node node, string detail) {
  // importlib.import_module
  exists(API::CallNode call |
    call =
      API::moduleImport("importlib").getMember("import_module").getACall() and
    node = call and
    detail = "importlib.import_module"
  )
  or
  // importlib.util.spec_from_file_location
  exists(API::CallNode call |
    call =
      API::moduleImport("importlib")
          .getMember("util")
          .getMember("spec_from_file_location")
          .getACall() and
    node = call and
    detail = "importlib.util.spec_from_file_location"
  )
  or
  // __import__
  exists(API::CallNode call |
    call = API::builtin("__import__").getACall() and
    node = call and
    detail = "__import__"
  )
}

// --- dotenv / .env loading ---
predicate isDotenvRead(DataFlow::Node node, string detail) {
  exists(API::CallNode call |
    (
      call = API::moduleImport("dotenv").getMember("load_dotenv").getACall() and
      detail = "dotenv.load_dotenv"
      or
      call = API::moduleImport("dotenv").getMember("dotenv_values").getACall() and
      detail = "dotenv.dotenv_values"
      or
      call = API::moduleImport("dotenv").getMember("find_dotenv").getACall() and
      detail = "dotenv.find_dotenv"
    ) and
    node = call
  )
}

// --- Credential file access ---
// Heuristic: open() with path containing credential-related keywords
predicate isCredentialFileRead(DataFlow::Node node, string detail) {
  exists(API::CallNode call, DataFlow::Node arg |
    call = API::builtin("open").getACall() and
    arg = call.getArg(0) and
    node = call and
    exists(string path |
      path = arg.asExpr().(StringLiteral).getText() and
      (
        path.matches("%credential%") or
        path.matches("%token%") or
        path.matches("%secret%") or
        path.matches("%api_key%") or
        path.matches("%apikey%") or
        path.matches("%.env%") or
        path.matches("%password%") or
        path.matches("%git-credentials%") or
        path.matches("%.ssh/%") or
        path.matches("%.aws/%") or
        path.matches("%.netrc%")
      ) and
      detail = "credential_file: " + path
    )
  )
}

// --- Process kill ---
// os.kill sends a signal to a process — this is the actual process_kill action.
// signal.signal (handler registration) is NOT process_kill; it's defensive, so excluded.
predicate isProcessKill(DataFlow::Node node, string detail) {
  exists(API::CallNode call |
    call = API::moduleImport("os").getMember("kill").getACall() and
    node = call and
    detail = "os.kill"
  )
}

// --- WebSocket (Python Concepts does not cover this) ---
predicate isWebSocket(DataFlow::Node node, string detail) {
  exists(API::CallNode call |
    (
      call = API::moduleImport("websockets").getMember("connect").getACall() or
      call = API::moduleImport("websocket").getMember("WebSocket").getACall() or
      call =
        API::moduleImport("websocket").getMember("WebSocketApp").getACall()
    ) and
    node = call and
    detail = "websocket connection"
  )
}

// --- DNS lookup ---
predicate isDnsLookup(DataFlow::Node node, string detail) {
  exists(API::CallNode call |
    (
      call =
        API::moduleImport("socket").getMember("getaddrinfo").getACall() or
      call =
        API::moduleImport("socket").getMember("gethostbyname").getACall() or
      call =
        API::moduleImport("dns")
            .getMember("resolver")
            .getMember("resolve")
            .getACall()
    ) and
    node = call and
    detail = "DNS lookup"
  )
}

// --- SMTP (email sending) ---
predicate isSmtpSend(DataFlow::Node node, string detail) {
  exists(API::CallNode call |
    (
      call = API::moduleImport("smtplib").getMember("SMTP").getACall() or
      call = API::moduleImport("smtplib").getMember("SMTP_SSL").getACall()
    ) and
    node = call and
    detail = "SMTP connection"
  )
}

// --- Cron/scheduling ---
predicate isCronSchedule(DataFlow::Node node, string detail) {
  exists(API::CallNode call |
    (
      call = API::moduleImport("crontab").getMember("CronTab").getACall() or
      call =
        API::moduleImport("schedule").getMember("every").getACall() or
      call =
        API::moduleImport("apscheduler")
            .getMember("schedulers")
            .getMember(["blocking", "background", "asyncio", "tornado", "twisted", "qt", "gevent"])
            .getMember(["BlockingScheduler", "BackgroundScheduler", "AsyncIOScheduler",
                         "TornadoScheduler", "TwistedScheduler", "QtScheduler", "GeventScheduler"])
            .getACall()
    ) and
    node = call and
    detail = "task scheduling"
  )
}

// --- Scrapling fetchers / sessions ---
predicate isScraplingFetch(DataFlow::Node node, string detail) {
  exists(API::CallNode call |
    (
      call =
        API::moduleImport("scrapling")
            .getMember("fetchers")
            .getMember(["Fetcher", "StealthyFetcher", "DynamicFetcher"])
            .getMember(["get", "post", "fetch", "request"])
            .getACall()
      or
      call =
        API::moduleImport("scrapling")
            .getMember("fetchers")
            .getMember(["FetcherSession", "StealthySession", "DynamicSession"])
            .getMember(["get", "post", "fetch", "request"])
            .getACall()
    ) and
    node = call and
    detail = "scrapling.fetchers"
  )
}

// ---------------------------------------------------------------------------
// Unified output (same schema as Tier 1)
// ---------------------------------------------------------------------------

from DataFlow::Node node, string capType, string detail
where
  (isEnvVarAccess(node, detail) and capType = "env_var_read")
  or
  (isDynamicImport(node, detail) and capType = "dynamic_import")
  or
  (isDotenvRead(node, detail) and capType = "token_file_read")
  or
  (isCredentialFileRead(node, detail) and capType = "token_file_read")
  or
  (isProcessKill(node, detail) and capType = "process_kill")
  or
  (isWebSocket(node, detail) and capType = "websocket")
  or
  (isDnsLookup(node, detail) and capType = "dns_lookup")
  or
  (isSmtpSend(node, detail) and capType = "smtp_send")
  or
  (isScraplingFetch(node, detail) and capType = "http_request")
  or
  (isCronSchedule(node, detail) and capType = "cron_schedule")
select node,
  "capType=" + capType + " | detail=" + detail + " | tier=api_graph"
