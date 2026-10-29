// SPDX-License-Identifier: BSD-3-Clause
//
// P02 Rocket/Gemmini/HardFloat linkage assertions.
//
// These checks run inside p02.Elaborate BEFORE elaboration. They prove that
// rocket-chip, cde, diplomacy, gemmini and berkeley-hardfloat all link into ONE
// classpath at the pinned versions (compile-time references below fail the build
// at sbt compile time; the Class.forName and construction checks fail elaboration
// at runtime with a nonzero exit).
package p02

import freechips.rocketchip.rocket.RocketCoreParams
import gemmini.GemminiConfigs

final case class CheckResult(name: String, passed: Boolean, detail: String)

object LinkCheck {

  /**
   * Typed class-presence check. Deliberately avoids Class.forName with string
   * names: several pinned symbols are NOT loadable by their source-dotted name
   * (e.g. CDE nests every API inside `object config`, so the binary name of
   * `org.chipsalliance.cde.config.Parameters` is `org.chipsalliance.cde.config$Parameters`;
   * Scala module classes end in `$`; rocket's old `freechips.rocketchip.diplomacy.LazyModule`
   * is only a compile-time type alias). `classOf`/`.getClass` resolve through
   * the compiler, and executing the reference here forces the class load.
   */
  private def typed(label: String, cls: => Class[_]): CheckResult =
    try {
      CheckResult(s"class:$label", passed = true, s"loaded as ${cls.getName}")
    } catch {
      case t: Throwable =>
        CheckResult(
          s"class:$label",
          passed = false,
          s"${t.getClass.getName}: ${if (t.getMessage == null) "" else t.getMessage}")
    }

  private def guard(name: String)(f: => CheckResult): CheckResult =
    try f
    catch {
      case t: Throwable =>
        CheckResult(
          name,
          passed = false,
          s"${t.getClass.getName}: ${if (t.getMessage == null) "" else t.getMessage}")
    }

  def run(): Seq[CheckResult] = {
    val b = Seq.newBuilder[CheckResult]

    // 1. Pinned classes resolve from the single merged classpath (typed refs).
    b += typed(
      "freechips.rocketchip.rocket.RocketCoreParams",
      classOf[freechips.rocketchip.rocket.RocketCoreParams])
    // Rocket's BuildRoCC Field (consumed by DefaultGemminiConfig): reference the
    // module instance; getClass forces the module class load.
    b += typed(
      "freechips.rocketchip.tile.BuildRoCC (module)",
      freechips.rocketchip.tile.BuildRoCC.getClass)
    // The rocket-side freechips.rocketchip.diplomacy.LazyModule name is only a
    // @deprecated type alias in the pinned shim (diplomacy/package.scala,
    // "LazyModule.scala" section) with no runtime class; the real class lives in
    // the standalone diplomacy library.
    b += typed(
      "org.chipsalliance.diplomacy.lazymodule.LazyModule",
      classOf[org.chipsalliance.diplomacy.lazymodule.LazyModule])
    // CDE keeps its API inside `object config`; the dotted source name resolves
    // via the compiler while the JVM binary name is config$Parameters.
    b += typed(
      "org.chipsalliance.cde.config.Parameters",
      classOf[org.chipsalliance.cde.config.Parameters])
    b += typed("gemmini.Gemmini", classOf[gemmini.Gemmini[_, _, _]])
    b += typed("hardfloat.MulAddRecFN", classOf[hardfloat.MulAddRecFN])

    // 2. Rocket constructs with its pinned RV64 + FPU + mul/div defaults.
    b += guard("rocket:RocketCoreParams()") {
      val p = RocketCoreParams()
      val ok = p.xLen == 64 && p.fpu.isDefined && p.mulDiv.isDefined
      CheckResult(
        "rocket:RocketCoreParams()",
        ok,
        s"xLen=${p.xLen} fpu=${p.fpu.isDefined} mulDiv=${p.mulDiv.isDefined}")
    }

    // 3. Gemmini default config object graph links; pinned 1x(16x16) int8 mesh
    //    with 32-bit accumulator as declared by GemminiConfigs.defaultConfig.
    b += guard("gemmini:GemminiConfigs.defaultConfig") {
      val cfg = GemminiConfigs.defaultConfig
      val ok =
        cfg.tileRows == 1 && cfg.tileColumns == 1 &&
          cfg.meshRows == 16 && cfg.meshColumns == 16 &&
          cfg.inputType.getWidth == 8 && cfg.accType.getWidth == 32
      CheckResult(
        "gemmini:GemminiConfigs.defaultConfig",
        ok,
        s"tile=${cfg.tileRows}x${cfg.tileColumns} mesh=${cfg.meshRows}x${cfg.meshColumns} " +
          s"inputBits=${cfg.inputType.getWidth} accBits=${cfg.accType.getWidth}")
    }

    // 4. Gemmini SInt arithmetic typeclass instance (used by DefaultGemminiConfig)
    //    resolves; compile-time reference, evaluated here to force class loading.
    b += guard("gemmini:Arithmetic.SIntArithmetic") {
      val _ = gemmini.Arithmetic.SIntArithmetic
      CheckResult("gemmini:Arithmetic.SIntArithmetic", passed = true, "typeclass instance resolved")
    }

    b.result()
  }
}
