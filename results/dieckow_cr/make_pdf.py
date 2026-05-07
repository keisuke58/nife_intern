#!/usr/bin/env python3
import subprocess, os, time, sys

env = dict(os.environ)
env['CHROME_PATH'] = '/home/nishioka/.cache/puppeteer/chrome/linux-131.0.6778.204/chrome-linux64/chrome'
env['DISPLAY'] = ':99'

xvfb = subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1280x1024x24'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)

here = os.path.dirname(os.path.abspath(__file__))
r = subprocess.run(
    ['/home/nishioka/.nvm/versions/node/v22.22.0/bin/npx',
     '@marp-team/marp-cli@latest',
     os.path.join(here, 'progress_report_hamilton_kegg.md'),
     '--pdf', '--allow-local-files',
     '-o', os.path.join(here, 'progress_report_hamilton_kegg.pdf')],
    env=env, capture_output=True, text=True, timeout=180,
    cwd=here
)
xvfb.terminate()
print('rc:', r.returncode)
print(r.stdout[-300:] if r.stdout else '')
print(r.stderr[-300:] if r.stderr else '', file=sys.stderr)
