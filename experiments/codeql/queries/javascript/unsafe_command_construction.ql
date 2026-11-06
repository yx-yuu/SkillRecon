/**
 * @name Source-bound unsafe shell command construction
 * @description Recover shell interpretation of unsanitized CLI or library inputs.
 * @kind path-problem
 * @problem.severity warning
 * @id skillrecon/javascript/unsafe-command-construction
 */

import javascript
import semmle.javascript.security.dataflow.UnsafeShellCommandConstructionCustomizations
import semmle.javascript.security.dataflow.UnsafeShellCommandConstructionQuery
import UnsafeShellCommandConstructionFlow::PathGraph

class SkillCommandLineInput extends UnsafeShellCommandConstruction::Source {
  SkillCommandLineInput() {
    exists(PropAccess argv |
      argv.getBase().(GlobalVarAccess).getName() = "process" and
      argv.getPropertyName() = "argv" and
      this = argv.flow()
    )
  }
}

from
  UnsafeShellCommandConstructionFlow::PathNode source,
  UnsafeShellCommandConstructionFlow::PathNode sink,
  UnsafeShellCommandConstruction::Sink shell
where
  UnsafeShellCommandConstructionFlow::flowPath(source, sink) and
  shell = sink.getNode()
select shell.getCommandExecution(), source, sink,
  "capType=shell_exec | tier=api_graph | detail=unsanitized input in shell command; " +
    shell.getSinkType() + "; source=" + source.getNode().getLocation().toString()
