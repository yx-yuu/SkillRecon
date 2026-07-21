/**
 * @name Import and call relationships
 * @description Extract cross-file import relationships for bridge detection.
 *              Covers both static imports and dynamic module loading patterns.
 * @kind problem
 * @problem.severity recommendation
 * @id skillrecon/python/call-imports
 */

import python

from ImportExpr imp, string importedModule
where importedModule = imp.getImportedModuleName()
select imp,
  "import_type=static | source_file=" +
    imp.getLocation().getFile().getBaseName() +
    " | imported=" + importedModule +
    " | tier=call_graph"
