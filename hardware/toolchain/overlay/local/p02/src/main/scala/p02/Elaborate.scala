// SPDX-License-Identifier: BSD-3-Clause
//
// P02 elaboration entry point.
//
// Usage:
//   p02.Elaborate <output-systemverilog-file> [--firtool <path>]
//
// argv(0) is the output SV path (created with parent directories). The optional
// --firtool flag overrides the environment in this priority:
//   --firtool flag  >  FIRTOOL_BIN  >  /opt/circt/bin/firtool
// The pinned image path /opt/circt/bin/firtool is the final fallback; bare PATH
// lookup is deliberately never used so the stage cannot silently pick up an
// unpinned firtool or attempt a network resolution.
//
// Exit codes: 0 success, 1 linkage/emit/write failure, 2 usage error.
package p02

import circt.stage.ChiselStage
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest

object Elaborate {

  private val pinnedFirtool = "/opt/circt/bin/firtool"

  private def usage(): String =
    Seq(
      "usage: p02.Elaborate <output-systemverilog-file> [--firtool <path>]",
      "  <output-systemverilog-file>  path of the emitted SystemVerilog file (required)",
      "  --firtool <path>             firtool binary (FIRTOOL_BIN > " +
        pinnedFirtool + ")",
    ).mkString(System.lineSeparator())

  private def die(msg: String, code: Int): Nothing = {
    System.err.println(s"p02.Elaborate: $msg")
    sys.exit(code)
  }

  def main(rawArgs: Array[String]): Unit = {
    var firtool = sys.env.getOrElse("FIRTOOL_BIN", pinnedFirtool)
    var outPath: Path = null
    val positional = Seq.newBuilder[String]

    var i = 0
    while (i < rawArgs.length) {
      rawArgs(i) match {
        case "--firtool" if i + 1 < rawArgs.length =>
          firtool = rawArgs(i + 1); i += 2
        case "--firtool" =>
          die("--firtool requires a path argument", 2)
        case "--help" | "-h" =>
          println(usage()); return
        case arg if arg.startsWith("-") =>
          die(s"unknown option '$arg'\n${usage()}", 2)
        case arg =>
          positional += arg; i += 1
      }
    }

    val pos = positional.result()
    if (pos.length != 1) {
      System.err.println(usage())
      sys.exit(2)
    }
    outPath = Paths.get(pos.head)

    // Explicit firtool must exist and be executable before anything runs.
    val fir = Paths.get(firtool)
    if (!Files.isRegularFile(fir) || !Files.isExecutable(fir)) {
      die(s"firtool binary not usable at '$firtool' (supply --firtool or set " +
        "FIRTOOL_BIN)", 1)
    }

    // Rocket/Gemmini/HardFloat linkage assertions first: a broken closure must
    // fail here, not inside firtool.
    val results = LinkCheck.run()
    results.foreach { r =>
      println(s"[linkage] ${if (r.passed) "PASS" else "FAIL"} ${r.name}: ${r.detail}")
    }
    if (results.exists(!_.passed)) {
      die("Rocket/Gemmini/HardFloat linkage assertions FAILED", 1)
    }

    // Clean elaboration through the pinned firtool; no implicit network access.
    val sv = ChiselStage.emitSystemVerilog(
      new HardFloatMulAddSmoke,
      args = Array("--firtool-binary-path", firtool),
      firtoolOpts = Array.empty,
    )

    if (sv == null || sv.isEmpty) {
      die("elaboration produced an empty SystemVerilog string", 1)
    }
    val hasTop = sv.contains("module HardFloatMulAddSmoke")
    val hasFma = sv.contains("MulAddRecFN")
    if (!hasTop || !hasFma) {
      die(s"emitted SV failed structure check (top=$hasTop mulAddRecFN=$hasFma)", 1)
    }

    val abs = outPath.toAbsolutePath
    val parent = abs.getParent
    if (parent != null) Files.createDirectories(parent)
    Files.write(abs, sv.getBytes(StandardCharsets.UTF_8))

    val bytes = Files.size(abs)
    val sha = MessageDigest
      .getInstance("SHA-256")
      .digest(sv.getBytes(StandardCharsets.UTF_8))
      .map(b => f"$b%02x")
      .mkString
    println(s"[p02.Elaborate] wrote $abs ($bytes bytes) firtool=$firtool sha256=$sha")
  }
}
