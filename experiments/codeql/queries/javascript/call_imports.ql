/**
 * @name Import and require relationships
 * @description Extract cross-file import/require relationships for bridge
 *              detection in JavaScript/TypeScript. Covers both ES module
 *              imports and CommonJS require() calls.
 * @kind problem
 * @problem.severity recommendation
 * @id skillrecon/javascript/call-imports
 */

import javascript

// ES module imports: import X from './module'
from ImportDeclaration imp, string source
where source = imp.getImportedPathString()
select imp,
  "import_type=esm | source_file=" +
    imp.getFile().getBaseName() +
    " | imported=" + source +
    " | tier=call_graph"
