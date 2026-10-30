// P03 numeric probe project — a fragment the parent appends VERBATIM to the
// tail of the fresh copied P02 overlay build.sbt (single build definition, no
// second .sbt compilation unit).
//
// The parent adapter copies hardware/toolchain/overlay (build.sbt + project/)
// to a fresh work folder, appends this fragment, and invokes:
//   sbt 'numericProbe/runMain p03.Elaborate <out.sv>'
//
// It only reuses definitions already present in build.sbt above the append
// point (commonSettings, chiselSettings, localBase, externalDir, hardfloat).
// The probe graph adds no new Scala dependencies and never touches the p02
// project.
//
// Environment:
//   NEXUS_NUMERIC_PROBE_SOURCE  source root containing NumericProbe.scala and
//                               Elaborate.scala (package p03). Required; the
//                               settings fail at evaluation otherwise.
lazy val numericProbe = (project in localBase("numericProbe"))
  .settings(commonSettings, chiselSettings)
  .settings(
    Compile / scalaSource := {
      val root = sys.env.getOrElse("NEXUS_NUMERIC_PROBE_SOURCE", "")
      if (root.trim.isEmpty) {
        throw new MessageOnlyException(
          "numericProbe: NEXUS_NUMERIC_PROBE_SOURCE must point at the p03 probe source root")
      }
      externalDir("numericProbe sources", root)
    },
  )
  .dependsOn(hardfloat)
