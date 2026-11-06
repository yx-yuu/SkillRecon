/** Canonical execution atoms and operands shared by events and taint paths. */
import javascript

string skillReconExecutionCapability(SystemCommandExecution execution) {
  exists(DataFlow::Node argument | execution.isShellInterpreted(argument)) and result = "shell_exec"
  or
  not exists(DataFlow::Node argument | execution.isShellInterpreted(argument)) and result = "subprocess"
}

predicate skillReconExecutionInput(SystemCommandExecution execution, DataFlow::Node input) {
  input = execution.getACommandArgument()
  or
  input = execution.getArgumentList()
}
