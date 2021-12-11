demucs -n mdx_extra /model/test/song_vocals_0311.mp3

python spliter.py --in_path /data/full_song \
  --out_separated_path /model/full_song_separated \
  --out_vocal_path /model/vocals

python gencqt.py --vocal_path /model/vocals \
  --out_vocals_path /model/database/vocals \
  --hum_path /data/hum \
  --out_hum_path /model/database/hum


python infer.py --load_model_path /model/weight/41_0.96.pth \
  --vocal_path  /model/database/vocals_npy \
  --hum_path  /model/database/hum_npy \
  --result_filename /result/submission.csv