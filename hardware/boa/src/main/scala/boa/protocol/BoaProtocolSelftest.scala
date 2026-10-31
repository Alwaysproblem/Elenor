// P10 BOA wire protocol selftest driver (Scala side).
//
// Consumes only the generated codec object boa.protocol.BoaProtocol:
//
//   roundtrip <in.bin> <out.bin>    decode every frame, re-encode, write out
//                                   and print "frames=N kinds=NAME:count,..."
//                                   (names sorted; the Python and C drivers
//                                   print the same string)
//   expect-fail <in.bin> <field>    decode must fail and name <field>
//
// Exit codes: 0 success, 1 unexpected success, 2 usage/IO error,
// 3 protocol rejection, 4 rejection names a different field.
package boa.protocol

import java.nio.file.{Files, Paths}

object BoaProtocolSelftest {

  private def countsLine(messages: Seq[BoaProtocol.BoaMessage]): String = {
    val counts = messages
      .groupBy(BoaProtocol.messageName)
      .toSeq
      .sortBy { case (name, _) => name }
    val rendered = counts.map { case (name, list) => s"$name:${list.size}" }.mkString(",")
    s"frames=${messages.size} kinds=$rendered"
  }

  private def roundtrip(inPath: String, outPath: String): Int = {
    val data = Files.readAllBytes(Paths.get(inPath))
    val messages = BoaProtocol.decodeStream(data)
    val out = BoaProtocol.encodeMany(messages)
    Files.write(Paths.get(outPath), out)
    println(countsLine(messages))
    0
  }

  private def expectSuccess(inPath: String): Int = {
    val data = Files.readAllBytes(Paths.get(inPath))
    BoaProtocol.decodeStream(data)
    System.err.println(s"unexpected-success: decode accepted $inPath")
    1
  }

  private def runExpectFail(inPath: String, field: String): Int =
    try expectSuccess(inPath)
    catch {
      case err: BoaProtocol.ProtocolError =>
        if (!err.field.contains(field)) {
          System.err.println(s"field=${err.field} reason=${err.reason}")
          System.err.println(s"expected field substring '$field' not named")
          4
        } else {
          println(s"rejected field=${err.field} reason=${err.reason}")
          0
        }
    }

  def main(args: Array[String]): Unit = {
    val code = try args.toList match {
      case List("roundtrip", in, out) => roundtrip(in, out)
      case List("expect-fail", in, field) => runExpectFail(in, field)
      case _ =>
        System.err.println("usage: roundtrip <in> <out> | expect-fail <in> <field>")
        2
    } catch {
      case err: BoaProtocol.ProtocolError =>
        System.err.println(s"field=${err.field} reason=${err.reason}")
        3
      case err: java.io.IOException =>
        System.err.println(s"io-error: ${err.getMessage}")
        2
    }
    if (code != 0) sys.exit(code)
  }
}
