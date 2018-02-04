import os, shutil

def make_dir(d):
    try:
      os.mkdir(d)
    except(FileExistsError):
      pass

# Empties the directories stocking the outputs or the program
def clear_dirs():
  folders = ['model', 'results']
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
