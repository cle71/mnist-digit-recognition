import os
import http.server
import socketserver
import webbrowser
from pathlib import Path

# Change to script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

PORT = 8001

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")

if __name__ == '__main__':
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"")
            print(f"=" * 60)
            print(f"MNIST Handwriting Recognition Server")
            print(f"=" * 60)
            print(f"Server running at: http://localhost:{PORT}")
            print(f"Open your browser and go to: http://localhost:{PORT}")
            print(f"")
            print(f"Instructions:")
            print(f"1. Draw a digit (0-9) on the canvas")
            print(f"2. Click 'Predict' button to recognize")
            print(f"3. Click 'Clear' to reset the canvas")
            print(f"")
            print(f"Press Ctrl+C to stop the server")
            print(f"=" * 60)
            print(f"")

            # Try to open browser automatically
            try:
                webbrowser.open(f'http://localhost:{PORT}')
            except Exception as e:
                print(f"Could not open browser automatically: {e}")

            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except OSError as e:
        print(f"Error: {e}")
        print(f"Port {PORT} might be in use. Try a different port.")
