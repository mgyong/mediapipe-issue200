"""Server for YouTube8M Model Inference Demo.

Serves up both the static files for the website and provides a service that
fetches the video id and timestamp based labels for a video analyzed in a
tfrecord files.

"""
import json
import os
import re
import socket
import subprocess

from absl import app
from absl import flags
import http.client
import http.server
from six.moves.urllib import parse

FLAGS = flags.FLAGS
flags.DEFINE_integer("port", 8008, "Port that the API is served over.")
flags.DEFINE_string("root", "/tmp/mediapipe", "Server location of assets.")
flags.DEFINE_string("pbtxt", "dataset_segment_level_inference.pbtxt",
                    "Default pbtxt file.")
flags.DEFINE_string("config", "static/config.json", "JSON Configuration.")


class HTTPServerV6(http.server.HTTPServer):
  address_family = socket.AF_INET6


class Youtube8MRequestHandler(http.server.SimpleHTTPRequestHandler):
  """Static file server with /healthz support."""

  def do_GET(self):
    if self.path.startswith("/healthz"):
      self.send_response(200)
      self.send_header("Content-type", "text/plain")
      self.send_header("Content-length", 2)
      self.end_headers()
      self.wfile.write("ok")
    if self.path.startswith("/video"):
      parsed_params = parse.urlparse(self.path)
      url_params = parse.parse_qs(parsed_params.query)

      index = 0
      tfrecord_path = ""

      if "file" in url_params:
        tfrecord_path = url_params["file"][0]
      if "index" in url_params:
        index = int(url_params["index"][0])

      self.fetch(tfrecord_path, index)

    else:
      if self.path == "/":
        self.path = "/index.html"
      # Default to serve up a local file
      self.path = "/static" + self.path
      http.server.SimpleHTTPRequestHandler.do_GET(self)

  def report_error(self, msg):
    """Simplifies sending out a string as a 500 http response."""
    self.send_response(500)
    self.send_header("Content-type", "text/plain")
    self.end_headers()
    self.wfile.write(bytes(msg, "utf-8"))

  def report_missing_files(self, files):
    """Sends out 500 response with missing files."""
    accumulate = ""
    for file_path in files:
      if not os.path.exists(file_path):
        accumulate = "%s '%s'" % (accumulate, file_path)

    if accumulate:
      self.report_error("Could not find:%s" % accumulate)
      return True

    return False

  def fetch(self, path, index):
    """Returns the video id and labels for a tfrecord at a provided index."""
    if (self.report_missing_files([
        "%s/%s" % (FLAGS.root, FLAGS.pbtxt),
        "%s/segment_level_inference" % FLAGS.root,
        "%s/label_map.txt" % FLAGS.root
    ])):
      return

    # Grab the TFRecord.
    filename_match = re.match("(train|validate|test)([^.]*)(\\.tfrecord)?",
                              path)
    if not filename_match:
      self.report_error("Filename '%s' is invalid." % path)
      return

    output_file = filename_match.expand(r"%s/\1\2.tfrecord" % FLAGS.root)
    tfrecord_url = filename_match.expand(
        r"http://us.data.yt8m.org/2/frame/\1/\1\2.tfrecord")

    connection = http.client.HTTPConnection("us.data.yt8m.org")
    connection.request("HEAD",
                       filename_match.expand(r"/2/frame/\1/\1\2.tfrecord"))
    response = connection.getresponse()
    if response.getheader("Content-Type") != "application/octet-stream":
      self.report_error("Filename '%s' is invalid." % path)

    if not os.path.exists(output_file):
      return_code = subprocess.call(
          ["curl", "--output", output_file, tfrecord_url],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE)
      if return_code:
        self.report_error("Could not retrieve contents from %s" % tfrecord_url)
        return

    process = subprocess.Popen([
        "%s/segment_level_inference" % FLAGS.root,
        "--calculator_graph_config_file=%s/" % FLAGS.root +
        "dataset_segment_level_inference.pbtxt",
        "--input_side_packets=tfrecord_path=%s,record_index_str=%d" %
        (output_file, index), "--output_stream=annotation_summary",
        "--output_stream_file=%s/labels" % FLAGS.root,
        "--output_side_packets=yt8m_id",
        "--output_side_packets_file=%s/yt8m_id" % FLAGS.root
    ],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout_str, stderr_str = process.communicate()
    process.wait()

    if stderr_str:
      self.report_error("Error executing segment_level_inference\n%s" %
                        stderr_str)

    f = open("%s/yt8m_id" % FLAGS.root, "r")
    contents = f.read()

    curl_arg = "data.yt8m.org/2/j/i/%s/%s.js" % (contents[-5:-3],
                                                 contents[-5:-1])
    process = subprocess.Popen(["curl", curl_arg],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout = process.communicate()
    process.wait()

    stdout_str = stdout[0].decode("utf-8")

    match = re.match(""".+"([^"]+)"[^"]+""", stdout_str)
    final_results = {
        "video_id": match.group(1),
        "link": "https://www.youtube.com/watch?v=%s" % match.group(1),
        "entries": []
    }
    f = open("%s/labels" % FLAGS.root, "r")
    lines = f.readlines()
    for line in lines:
      entry = {"labels": []}
      final_results["entries"].append(entry)
      first = True
      for column in line.split(","):
        if first:
          entry["time"] = float(int(column)) / 1000000.0
          first = False
        else:
          label_score = re.match("(.+):([0-9.]+).*", column)
          score = float(label_score.group(2))
          entry["labels"].append({
              "label": label_score.group(1),
              "score": score
          })
    response_json = json.dumps(final_results, indent=2, separators=(",", ": "))
    self.send_response(200)
    self.send_header("Content-type", "application/json")
    self.end_headers()
    self.wfile.write(bytes(response_json, "utf-8"))


def main(unused_args):
  dname = os.path.dirname(os.path.abspath(__file__))
  os.chdir(dname)
  port = FLAGS.port
  print("Listening on port %s" % port)  # pylint: disable=superfluous-parens
  server = HTTPServerV6(("::", int(port)), Youtube8MRequestHandler)
  server.serve_forever()


if __name__ == "__main__":
  app.run(main)
