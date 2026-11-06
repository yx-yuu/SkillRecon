/** Canonical execution atom shared by capability and payload queries. */
import python
import semmle.python.ApiGraphs
import semmle.python.dataflow.new.DataFlow

predicate skillReconSubprocessInvocation(DataFlow::Node node) {
  node = API::moduleImport("subprocess").getMember([
      "run", "Popen", "call", "check_call", "check_output"
    ]).getACall()
}

string skillReconExecutionCapability(DataFlow::Node node) {
  skillReconSubprocessInvocation(node) and result = "subprocess"
  or
  not skillReconSubprocessInvocation(node) and result = "shell_exec"
}
