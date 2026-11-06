/** File access kinds and path operands shared by capability and flow queries. */
import python
import semmle.python.ApiGraphs
import semmle.python.Concepts
import semmle.python.dataflow.new.DataFlow

predicate skillReconOpen(API::CallNode call) {
  call = API::builtin("open").getACall()
  or call = API::moduleImport("io").getMember("open").getACall()
  or call = API::moduleImport("codecs").getMember("open").getACall()
}

predicate skillReconWriteOpen(DataFlow::Node node) {
  exists(API::CallNode call, string mode |
    skillReconOpen(call) and node = call and
    mode = [call.getArg(1), call.getArgByName("mode")].getALocalSource().asExpr().(StringLiteral).getText() and
    mode.regexpMatch(".*[wax+].*")
  )
}

predicate skillReconReadAccess(FileSystemAccess access) {
  not access instanceof FileSystemWriteAccess and
  not exists(API::CallNode call, string mode |
    skillReconOpen(call) and access = call and
    mode = [call.getArg(1), call.getArgByName("mode")].getALocalSource().asExpr().(StringLiteral).getText() and
    mode.regexpMatch(".*[wax].*") and not mode.matches("%+%")
  ) and
  not exists(API::CallNode mutation |
    mutation = API::moduleImport("os").getMember(["remove", "unlink", "rmdir", "mkdir", "makedirs", "chmod", "chown", "rename", "replace"]).getACall()
    and access = mutation
  )
}

string skillReconFileResource(DataFlow::Node node) {
  result = node.(FileSystemAccess).getAPathArgument().getALocalSource().asExpr().(StringLiteral).getText()
}
