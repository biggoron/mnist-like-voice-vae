import os, shutil

folders = ['models', 'results']

for f in folders:
  for fi in os.listdir(f):
    fp = os.path.join(f, fi)
    try:
      if os.path.isfile(fp):
        os.unlink(fp)
      elif os.path.isdir(fp):
        shutil.rmtree(fp)
    except Exception as e:
      print(e)
