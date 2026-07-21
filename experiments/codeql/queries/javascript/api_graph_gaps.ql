/**
 * @name Skill capability events (Tier 2 — API gaps)
 * @description Detect behaviors NOT covered by CodeQL Concepts but important
 *              for SkillRecon's capability taxonomy in JavaScript/TypeScript.
 * @kind problem
 * @problem.severity recommendation
 * @id skillrecon/javascript/api-graph-gaps
 */

import javascript
import semmle.javascript.ES2015Modules
import semmle.javascript.NodeJS

predicate isImportedModuleVar(VarAccess access, string modulePath) {
  exists(ImportDefaultSpecifier spec |
    spec.getLocal().getName() = access.getName() and
    spec.getImportDeclaration().getRawImportPath() = modulePath
  )
  or
  exists(ImportNamespaceSpecifier spec |
    spec.getLocal().getName() = access.getName() and
    spec.getImportDeclaration().getRawImportPath() = modulePath
  )
  or
  exists(VariableDeclarator decl, Require req |
    decl.getBindingPattern().getAVariable().getName() = access.getName() and
    req = decl.getInit().(Require) and
    req.getArgument(0).(StringLiteral).getStringValue() = modulePath
  )
}

predicate isNamedImportVar(VarAccess access, string modulePath, string importedName) {
  exists(NamedImportSpecifier spec |
    spec.getLocal().getName() = access.getName() and
    spec.getImportedName() = importedName and
    spec.getImportDeclaration().getRawImportPath() = modulePath
  )
}

predicate isImportedFunction(InvokeExpr call, string modulePath, string importedName) {
  exists(NamedImportSpecifier spec |
    call.getCalleeName() = spec.getLocal().getName() and
    spec.getImportedName() = importedName and
    spec.getImportDeclaration().getRawImportPath() = modulePath
  )
}

// ---------------------------------------------------------------------------
// Tier 2: Gap fillers for JS/TS
// ---------------------------------------------------------------------------

// --- Environment variable access ---
// process.env.KEY or process.env["KEY"]
predicate isEnvVarAccess(DataFlow::Node node, string detail) {
  exists(PropAccess pa |
    pa.getBase().(PropAccess).getPropertyName() = "env" and
    pa.getBase().(PropAccess).getBase().(GlobalVarAccess).getName() = "process" and
    node = pa.flow() and
    detail = "process.env." + pa.getPropertyName()
  )
}

// --- Dynamic import / require ---
predicate isDynamicImport(DataFlow::Node node, string detail) {
  // Dynamic require with non-literal argument
  exists(CallExpr call |
    call.getCalleeName() = "require" and
    not call.getArgument(0) instanceof StringLiteral and
    node = call.flow() and
    detail = "dynamic require()"
  )
  or
  // Dynamic import() expression
  exists(DynamicImportExpr imp |
    node = imp.flow() and
    detail = "dynamic import()"
  )
}

// --- eval/Function constructor (JS CodeExecution gaps) ---
predicate isCodeExecution(DataFlow::Node node, string detail) {
  exists(CallExpr call |
    call.getCalleeName() = "eval" and
    node = call.flow() and
    detail = "eval()"
  )
  or
  exists(NewExpr ne |
    ne.getCalleeName() = "Function" and
    node = ne.flow() and
    detail = "new Function()"
  )
}

// --- dotenv / .env loading ---
predicate isDotenvConfig(DataFlow::Node node, string detail) {
  exists(MethodCallExpr call |
    call.getMethodName() = "config" and
    (
      exists(Require req |
        req = call.getReceiver() and
        req.getArgument(0).(StringLiteral).getStringValue() = "dotenv"
      )
      or
      exists(VarAccess receiver |
        receiver = call.getReceiver() and
        isImportedModuleVar(receiver, "dotenv")
      )
    ) and
    node = call.flow() and
    detail = "dotenv.config"
  )
  or
  exists(CallExpr call |
    isImportedFunction(call, "dotenv", "config") and
    node = call.flow() and
    detail = "dotenv.config"
  )
}

// --- viem / ethers RPC client setup ---
predicate isBlockchainRpcClient(DataFlow::Node node, string detail) {
  exists(CallExpr call |
    (
      isImportedFunction(call, "viem", "createPublicClient") and
      detail = "viem.createPublicClient"
      or
      isImportedFunction(call, "viem", "createWalletClient") and
      detail = "viem.createWalletClient"
      or
      isImportedFunction(call, "viem", "http") and
      detail = "viem.http"
    ) and
    node = call.flow()
  )
  or
  exists(NewExpr ne, PropAccess callee, VarAccess base |
    callee = ne.getCallee() and
    base = callee.getBase() and
    callee.getPropertyName() = "JsonRpcProvider" and
    (
      isImportedModuleVar(base, "ethers")
      or
      isNamedImportVar(base, "ethers", "ethers")
    ) and
    node = ne.flow() and
    detail = "ethers.JsonRpcProvider"
  )
  or
  exists(NewExpr ne |
    isImportedFunction(ne, "ethers", "JsonRpcProvider") and
    node = ne.flow() and
    detail = "ethers.JsonRpcProvider"
  )
}

from DataFlow::Node node, string capType, string detail
where
  (isEnvVarAccess(node, detail) and capType = "env_var_read")
  or
  (isDynamicImport(node, detail) and capType = "dynamic_import")
  or
  (isCodeExecution(node, detail) and capType = "eval_exec")
  or
  (isDotenvConfig(node, detail) and capType = "token_file_read")
  or
  (isBlockchainRpcClient(node, detail) and capType = "http_request")
select node,
  "capType=" + capType + " | detail=" + detail + " | tier=api_graph"
