/**
 * @name Skill source-to-sink paths
 * @description Recover minimal source-to-sink paths for SkillRecon using
 *              CodeQL taint tracking. This query is intentionally scoped to
 *              the most critical source/sink families required by M2.
 * @kind path-problem
 * @problem.severity warning
 * @id skillrecon/python/sensitive-paths
 */

import python
import semmle.python.Concepts
import semmle.python.ApiGraphs
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking
import SkillReconPathlib
import SkillReconFileIO
import SkillReconExecution

predicate isEnvSource(DataFlow::Node source) {
  exists(API::CallNode call |
    call = API::moduleImport("os").getMember(["getenv", "getenvb"]).getACall() and
    source = call
  )
  or
  exists(API::CallNode call |
    call =
      API::moduleImport("os").getMember("environ").getMember("get").getACall() and
    source = call
  )
  or
  exists(API::Node sub |
    sub = API::moduleImport("os").getMember("environ").getASubscript() and
    source = sub.asSource()
  )
}

predicate getSourceSlotKind(DataFlow::Node source, string slotKind) {
  isEnvSource(source) and slotKind = "env_access"
  or
  isFileSource(source) and slotKind = "file_read_result"
}

predicate isFileSource(DataFlow::Node source) {
  exists(FileSystemAccess access |
    skillReconReadAccess(access) and
    source = access
  )
  or
  skillReconPathlibRead(source)
}

predicate isShellSink(DataFlow::Node sink) {
  exists(SystemCommandExecution exec |
    sink = exec.getCommand()
  )
}

predicate isFileWriteSink(DataFlow::Node sink) {
  exists(FileSystemWriteAccess write |
    sink = write.getADataNode()
  )
}

predicate httpBodyCall(API::CallNode call, DataFlow::Node sink) {
  exists(string methodName |
    methodName in [Http::httpVerbLower(), "request"] and
    (
      call = API::moduleImport("requests").getMember(methodName).getACall()
      or
      call = API::moduleImport("httpx").getMember(methodName).getACall()
    ) and
    (
      exists(string argName |
        argName in ["data", "json", "content", "body", "params", "headers"] and
        sink = call.getArgByName(argName)
      )
      or
      methodName = "request" and sink = call.getArg(2)
      or
      not methodName = "request" and sink = call.getArg(1)
    )
  )
}

predicate isHttpBodySink(DataFlow::Node sink) { httpBodyCall(_, sink) }

predicate isHttpUrlSink(DataFlow::Node sink) {
  exists(Http::Client::Request request |
    sink = request.getAUrlPart()
  )
}

predicate getSinkSlotKind(DataFlow::Node sink, string slotKind) {
  isShellSink(sink) and slotKind = "command_arg"
  or
  isFileWriteSink(sink) and slotKind = "file_write_arg"
  or
  isHttpBodySink(sink) and slotKind = "http_body_arg"
  or
  isHttpUrlSink(sink) and slotKind = "http_url_arg"
}

predicate getSourceCapability(DataFlow::Node source, string capType) {
  isEnvSource(source) and capType = "env_var_read"
  or
  isFileSource(source) and capType = "file_read"
}

predicate getSinkCapability(DataFlow::Node sink, string capType) {
  exists(SystemCommandExecution exec |
    sink = exec.getCommand() and capType = skillReconExecutionCapability(exec)
  )
  or
  isFileWriteSink(sink) and capType = "file_write"
  or
  (isHttpBodySink(sink) or isHttpUrlSink(sink)) and capType = "http_request"
}

string getSimpleNodeSymbolHint(DataFlow::Node node) {
  result = node.asExpr().(Name).getId()
  or
  not exists(node.asExpr().(Name).getId()) and
  result = ""
}

predicate assignedSymbolHint(DataFlow::Node node, string hint) {
  exists(AssignStmt assign, Name lhs |
    assign.getValue() = node.asExpr() and
    assign.getATarget() = lhs and
    not exists(Expr other | assign.getATarget() = other and other != lhs) and
    hint = lhs.getId()
  )
  or
  exists(AssignStmt assign, Name lhs |
    assign.getValue().getAFlowNode() = node.asCfgNode() and
    assign.getATarget() = lhs and
    not exists(Expr other | assign.getATarget() = other and other != lhs) and
    hint = lhs.getId()
  )
}

string getAssignedSymbolHint(DataFlow::Node node) {
  assignedSymbolHint(node, result)
  or
  not exists(string hint | assignedSymbolHint(node, hint)) and
  result = ""
}

string getNodeSite(DataFlow::Node node) {
  result = node.getLocation().getFile().getRelativePath() + ":" +
    node.getLocation().getStartLine().toString() + ":" +
    node.getLocation().getStartColumn().toString()
}

string getSinkOperationSite(DataFlow::Node sink, string cap) {
  cap in ["shell_exec", "subprocess"] and
  exists(SystemCommandExecution exec | sink = exec.getCommand() and result = getNodeSite(exec))
  or
  cap = "file_write" and
  exists(FileSystemWriteAccess write | sink = write.getADataNode() and result = getNodeSite(write))
  or
  cap = "http_request" and
  (
    exists(API::CallNode call | httpBodyCall(call, sink) and result = getNodeSite(call))
    or
    exists(Http::Client::Request request | sink = request.getAUrlPart() and result = getNodeSite(request))
  )
}

module SkillReconPathConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) { getSourceCapability(source, _) }

  predicate isSink(DataFlow::Node sink) { getSinkCapability(sink, _) }

  // Authorization ordering follows derived values, including fingerprints.
  // A digest is a dependency on its input; this does not assert reversibility
  // or that the original token bytes are disclosed.
  predicate isAdditionalFlowStep(DataFlow::Node input, DataFlow::Node output) {
    exists(API::CallNode hash, API::CallNode digest |
      hash = API::moduleImport("hashlib").getMember("sha256").getACall() and
      digest = hash.getReturn().getMember(["digest", "hexdigest"]).getACall() and
      input = hash.getArg(0) and output = digest
    )
  }
}

module SkillReconPathFlow = TaintTracking::Global<SkillReconPathConfig>;
import SkillReconPathFlow::PathGraph

from
  SkillReconPathFlow::PathNode source,
  SkillReconPathFlow::PathNode sink,
  string sourceCap,
  string sinkCap,
  string sourceSlotKind,
  string sinkSlotKind,
  string sourceSymbolHint,
  string sinkSymbolHint
where
  SkillReconPathFlow::flowPath(source, sink) and
  getSourceCapability(source.getNode(), sourceCap) and
  getSinkCapability(sink.getNode(), sinkCap) and
  getSourceSlotKind(source.getNode(), sourceSlotKind) and
  getSinkSlotKind(sink.getNode(), sinkSlotKind) and
  sourceSymbolHint = getAssignedSymbolHint(source.getNode()) and
  sinkSymbolHint = getSimpleNodeSymbolHint(sink.getNode())
select sink.getNode(), source, sink,
  "pathSource=" + sourceCap +
    " | sourceSlotKind=" + sourceSlotKind +
    " | sourceSymbolHint=" + sourceSymbolHint +
    " | pathSink=" + sinkCap +
    " | sinkSlotKind=" + sinkSlotKind +
    " | sinkSymbolHint=" + sinkSymbolHint +
    " | sourceSite=" + getNodeSite(source.getNode()) +
    " | sinkSite=" + getNodeSite(sink.getNode()) +
    " | sourceOperationSite=" + getNodeSite(source.getNode()) +
    " | sinkOperationSite=" + getSinkOperationSite(sink.getNode(), sinkCap) +
    " | pathKind=codeql:python:" + sourceCap + "->" + sinkCap
