/**
 * @name Skill source-to-sink paths
 * @description Recover minimal source-to-sink paths for SkillRecon in
 *              JavaScript/TypeScript using CodeQL taint tracking.
 * @kind path-problem
 * @problem.severity warning
 * @id skillrecon/javascript/sensitive-paths
 */

import javascript
import SkillReconExecution
import semmle.javascript.dataflow.DataFlow
import semmle.javascript.dataflow.TaintTracking
import semmle.javascript.frameworks.WebSocket

predicate isEnvSource(DataFlow::Node source) {
  exists(PropAccess pa |
    pa.getBase().(PropAccess).getPropertyName() = "env" and
    pa.getBase().(PropAccess).getBase().(GlobalVarAccess).getName() = "process" and
    source = pa.flow()
  )
}

predicate getSourceSlotKind(DataFlow::Node source, string slotKind) {
  isEnvSource(source) and slotKind = "env_access"
  or
  isFileSource(source) and slotKind = "file_read_result"
}

predicate isFileSource(DataFlow::Node source) {
  exists(FileSystemReadAccess read |
    source = read.getADataNode().getALocalSource()
  )
}

predicate isShellSink(DataFlow::Node sink) {
  exists(SystemCommandExecution exec |
    skillReconExecutionInput(exec, sink)
  )
}

predicate isFileWriteSink(DataFlow::Node sink) {
  exists(FileSystemWriteAccess write |
    sink = write.getADataNode()
  )
}

predicate isHttpSink(DataFlow::Node sink) {
  exists(ClientRequest request |
    not request instanceof ClientWebSocket::ClientSocket and
    (
      sink = request.getUrl()
      or
      sink = request.getADataNode()
    )
  )
}

predicate isWebSocketSink(DataFlow::Node sink) {
  exists(ClientWebSocket::ClientSocket socket |
    sink = socket.getUrl()
    or
    sink = socket.getADataNode()
  )
}

predicate getSinkSlotKind(DataFlow::Node sink, string slotKind) {
  isWebSocketSink(sink) and slotKind = "http_url_arg"
  or
  isShellSink(sink) and slotKind = "command_arg"
  or
  isFileWriteSink(sink) and slotKind = "file_write_arg"
  or
  exists(ClientRequest request |
    not request instanceof ClientWebSocket::ClientSocket and
    sink = request.getUrl() and
    slotKind = "http_url_arg"
  )
  or
  exists(ClientRequest request |
    not request instanceof ClientWebSocket::ClientSocket and
    sink = request.getADataNode() and
    slotKind = "http_body_arg"
  )
}

predicate getSourceCapability(DataFlow::Node source, string capType) {
  isEnvSource(source) and capType = "env_var_read"
  or
  isFileSource(source) and capType = "file_read"
}

predicate getSinkCapability(DataFlow::Node sink, string capType) {
  isWebSocketSink(sink) and capType = "websocket"
  or
  exists(SystemCommandExecution exec |
    skillReconExecutionInput(exec, sink) and capType = skillReconExecutionCapability(exec)
  )
  or
  isFileWriteSink(sink) and capType = "file_write"
  or
  isHttpSink(sink) and capType = "http_request"
}

string getEnvSourceSymbolHint(DataFlow::Node source) {
  result = source.asExpr().(PropAccess).getPropertyName()
  or
  not exists(source.asExpr().(PropAccess).getPropertyName()) and
  result = ""
}

string getSimpleNodeSymbolHint(DataFlow::Node node) {
  result = node.asExpr().(VarAccess).getName()
  or
  not exists(node.asExpr().(VarAccess).getName()) and
  result = ""
}

string getNodeSite(DataFlow::Node node) {
  result = node.getLocation().getFile().getRelativePath() + ":" +
    node.getLocation().getStartLine().toString() + ":" +
    node.getLocation().getStartColumn().toString()
}

string getSourceOperationSite(DataFlow::Node source, string cap) {
  cap = "env_var_read" and result = getNodeSite(source)
  or
  cap = "file_read" and
  exists(FileSystemReadAccess read |
    source = read.getADataNode().getALocalSource() and result = getNodeSite(read)
  )
}

string getSinkOperationSite(DataFlow::Node sink, string cap) {
  exists(SystemCommandExecution exec |
    skillReconExecutionInput(exec, sink) and cap = skillReconExecutionCapability(exec) and
    result = getNodeSite(exec)
  )
  or
  cap = "file_write" and
  exists(FileSystemWriteAccess write | sink = write.getADataNode() and result = getNodeSite(write))
  or
  cap = "http_request" and
  exists(ClientRequest request |
    not request instanceof ClientWebSocket::ClientSocket and
    (sink = request.getUrl() or sink = request.getADataNode()) and result = getNodeSite(request)
  )
  or
  cap = "websocket" and
  exists(ClientWebSocket::ClientSocket socket |
    (sink = socket.getUrl() or sink = socket.getADataNode()) and result = getNodeSite(socket)
  )
}

module SkillReconPathConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) { getSourceCapability(source, _) }

  predicate isSink(DataFlow::Node sink) { getSinkCapability(sink, _) }
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
  sourceSymbolHint = getEnvSourceSymbolHint(source.getNode()) and
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
    " | sourceOperationSite=" + getSourceOperationSite(source.getNode(), sourceCap) +
    " | sinkOperationSite=" + getSinkOperationSite(sink.getNode(), sinkCap) +
    " | pathKind=codeql:javascript:" + sourceCap + "->" + sinkCap
