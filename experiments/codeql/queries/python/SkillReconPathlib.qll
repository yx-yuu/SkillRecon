/** Shared registered pathlib read-result model for events and payload sources. */
import python
import semmle.python.ApiGraphs
import semmle.python.dataflow.new.DataFlow

predicate skillReconPathlibRead(DataFlow::Node node) {
  exists(API::Node instance, API::CallNode call |
    (
      instance = API::moduleImport("pathlib").getMember("Path").getReturn()
      or
      instance =
        API::moduleImport("pathlib").getMember("Path").getReturn()
          .getMember("expanduser").getReturn()
    ) and
    call = instance.getMember(["read_text", "read_bytes"]).getACall() and
    node = call
  )
}
