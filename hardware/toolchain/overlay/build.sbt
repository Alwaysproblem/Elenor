// P02 toolchain overlay — sbt 1.8.2 / Scala 2.13.12 / Chisel 6.5.0 (org.chipsalliance)
//
// This build file replicates the authoritative Chipyard sbt graph (Chipyard pin
// e02074414158c85059c1dcb231ff2f8941254b9a, build.sbt of that commit) for the
// minimal HardFloat/Rocket/Gemmini source closure, with two P02-mandated
// structural changes:
//
//   1. Every project base lives under <overlay>/local/<id> so sbt writes all
//      outputs (target/, etc.) to a writable, freshly-copied location. The
//      upstream sources stay read-only and are attached via
//      `Compile / scalaSource` (and the analogous resourceDirectory).
//   2. Gemmini is compiled from the single authoritative NEXUS_GEMMINI source
//      root (default /workspace/hardware/gemmini/upstream); the Chipyard-internal
//      generators/gemmini checkout is never referenced.
//
// The upstream subprojects' own build definitions (berkeley-hardfloat build.sbt
// with chisel3 3.5.6, gemmini standalone build.sbt with 2.13.10/3.6.0, rocket-chip
// Mill build.sc) are NOT loaded: sbt only evaluates this root build.sbt. This is
// the same property Chipyard itself relies on.
//
// Environment:
//   NEXUS_DEPS      Chipyard pinned dependency root (default /opt/deps/chipyard)
//   NEXUS_GEMMINI   Gemmini upstream root (default /workspace/hardware/gemmini/upstream)
//   FIRTOOL_BIN     firtool consumed by p02.Elaborate (falls back to /opt/circt/bin/firtool)
//
// Commands:
//   sbt p02/compile                                   compile whole pinned closure + p02
//   sbt 'p02/runMain p02.Elaborate <out.sv>'           clean elaborate (explicit firtool path inside)
//
// Resolution: default Maven Central only. Chipyard also declares sonatype
// snapshot/release resolvers and mavenLocal; those are deliberately dropped so
// offline resolution is served exclusively by the packaged sbt bundle
// (/opt/sbt-bundle) and the Maven-central-only repository file. Every pinned
// artifact below resolves from Maven Central.

// Pins replicated from Chipyard build.sbt at e02074414158c85059c1dcb231ff2f8941254b9a
val chiselVersion = "6.5.0"
val scalaPin      = "2.13.12"

// Chipyard root env: NEXUS_DEPS is the frozen shared-contract name.
val depsRoot    = sys.env.getOrElse("NEXUS_DEPS", "/opt/deps/chipyard")
val gemminiRoot = sys.env.getOrElse("NEXUS_GEMMINI", "/workspace/hardware/gemmini/upstream")

/** Fail the build at load time if a pinned external source root is missing. */
def externalDir(label: String, path: String): File = {
  val f = file(path)
  if (!f.isDirectory) {
    sys.error(
      s"P02 overlay: pinned $label source root not found: $path " +
        "(set NEXUS_DEPS / NEXUS_GEMMINI to the pinned dependency roots)")
  }
  f
}

/** Writable local project base under the overlay (target/ output lands here). */
def localBase(id: String): File = file("local") / id

/**
 * Project with a writable local base whose Scala (and resource) sources are the
 * pinned external read-only roots. resourceDir mirrors Chipyard's freshProject
 * semantics: freshProject uses baseDirectory = <dir>/src, so its resource
 * directory is <dir>/src/main/resources; plain `(project in dir)` projects keep
 * <dir>/src/main/resources.
 */
def pinnedProject(id: String, scalaDir: String, resourceDir: String): Project =
  Project(id = id, base = localBase(id))
    .settings(
      Compile / scalaSource := externalDir(id + " sources", scalaDir),
      Compile / resourceDirectory := file(resourceDir),
    )

lazy val commonSettings = Seq(
  organization := "edu.berkeley.cs",
  version := "1.6",
  scalaVersion := scalaPin,
  scalacOptions ++= Seq(
    "-deprecation",
    "-unchecked",
    "-Ytasty-reader",
    "-Ymacro-annotations"), // fix hierarchy API
  libraryDependencies += "com.lihaoyi" %% "sourcecode" % "0.3.1",
  libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value,
  exportJars := true,
)

lazy val chiselSettings = Seq(
  libraryDependencies += "org.chipsalliance" %% "chisel" % chiselVersion,
  addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full),
  libraryDependencies ++= Seq(
    "org.apache.commons" % "commons-lang3" % "3.12.0",
    "org.apache.commons" % "commons-text" % "1.9",
  ),
)

// ---------------------------------------------------------------------------
// Minimal pinned source closure (compile order: midas_target_utils -> hardfloat;
// cde -> diplomacy; rocketMacros; rocketchip; gemmini; p02)
// ---------------------------------------------------------------------------

// FireSim sim/midas/targetutils, pin b084672c2f8cf32e55d78f73a001074c23f8a2b8
lazy val midas_target_utils = pinnedProject(
    "midas_target_utils",
    s"$depsRoot/sims/firesim/sim/midas/targetutils/src/main/scala",
    s"$depsRoot/sims/firesim/sim/midas/targetutils/src/main/resources",
  )
  .settings(commonSettings, chiselSettings)

// Berkeley HardFloat, pin 4225367ed282aadd70f4cbd131f6959f4d00c502
// (sources in the nested hardfloat/ subdirectory, as in Chipyard)
lazy val hardfloat = pinnedProject(
    "hardfloat",
    s"$depsRoot/generators/hardfloat/hardfloat/src/main/scala",
    s"$depsRoot/generators/hardfloat/hardfloat/src/main/resources",
  )
  .settings(commonSettings, chiselSettings)
  .dependsOn(midas_target_utils)

// Rocket-chip Scala macros, pin c7aabd2823d3d001a7a9e8a982bb65d6cb7c9bdf
lazy val rocketMacros = pinnedProject(
    "rocketMacros",
    s"$depsRoot/generators/rocket-chip/macros/src/main/scala",
    s"$depsRoot/generators/rocket-chip/macros/src/main/resources",
  )
  .settings(commonSettings)

// CDE, pin 2bcaeae2b9914bd25497ce3c6fa62dc5ca80e09f
// (nonstandard source root cde/src/chipsalliance/rocketchip, as in Chipyard)
lazy val cde = pinnedProject(
    "cde",
    s"$depsRoot/tools/cde/cde/src/chipsalliance/rocketchip",
    s"$depsRoot/tools/cde/src/main/resources",
  )
  .settings(commonSettings)

// Diplomacy, pin 6b7dc988a771ff21237b102fd02a1f2d2c2e7b72
// (nonstandard source root src/diplomacy, as in Chipyard)
lazy val diplomacy = pinnedProject(
    "diplomacy",
    s"$depsRoot/generators/diplomacy/diplomacy/src/diplomacy",
    s"$depsRoot/generators/diplomacy/diplomacy/src/main/resources",
  )
  .settings(commonSettings, chiselSettings)
  .dependsOn(cde)

// Rocket Chip, pin c7aabd2823d3d001a7a9e8a982bb65d6cb7c9bdf
lazy val rocketchip = pinnedProject(
    "rocketchip",
    s"$depsRoot/generators/rocket-chip/src/main/scala",
    s"$depsRoot/generators/rocket-chip/src/main/resources",
  )
  .settings(commonSettings, chiselSettings)
  .settings(
    libraryDependencies ++= Seq(
      "com.lihaoyi"   %% "mainargs"       % "0.5.0",
      "org.json4s"    %% "json4s-jackson" % "4.0.5",
      "org.scala-graph" %% "graph-core"   % "1.13.5",
    ),
  )
  .dependsOn(hardfloat, rocketMacros, diplomacy, cde)

// Gemmini, pin 25809f78323a729ef76fb68f3cedd8a24da2942b — the ONLY authoritative
// Gemmini source root (NEXUS_GEMMINI). Gemmini's standalone build.sbt (sbt 1.3.13,
// Scala 2.13.10, chisel3 3.6.0, floating rocketchip 1.2.+) must never be loaded.
// Chipyard applies rocketLibDeps here; applying chiselSettings + dependsOn(rocketchip)
// supplies the same compile classpath (chisel/plugin plus rocket's library deps
// arrive via the inter-project dependency).
lazy val gemmini = pinnedProject(
    "gemmini",
    s"$gemminiRoot/src/main/scala",
    s"$gemminiRoot/src/main/resources",
  )
  .settings(commonSettings, chiselSettings)
  .dependsOn(rocketchip)

// P02 elaboration + linkage assertions
lazy val p02 = (project in localBase("p02"))
  .settings(commonSettings, chiselSettings)
  .dependsOn(hardfloat, rocketchip, gemmini)

// ---------------------------------------------------------------------------
// Root: aggregation only, no sources of its own.
// ---------------------------------------------------------------------------

lazy val p02OverlayRoot = (project in file("."))
  .aggregate(midas_target_utils, hardfloat, rocketMacros, cde, diplomacy, rocketchip, gemmini, p02)
  .settings(
    publish / skip := true,
  )

// Dependency-inventory note for toolchain.lock: parent hashes the packaged
// /opt/sbt-bundle directly; no inventory command aliases are defined here.
// Runtime elaboration is invoked as `sbt 'p02/runMain p02.Elaborate <abs sv>'`.
