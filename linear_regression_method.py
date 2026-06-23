from data_process_method import *

song_info = pd.read_csv("spotify_data.csv")
user_hist = pd.read_csv("spotify_history.csv")
merged = pd.merge(user_hist, song_info, on='track_name')

merged = merged.dropna(subset="track_genre")
merged["explicit"] = merged["explicit"].map({True: 1, False: 0})
agg_dict = {col: "first" for col in merged.columns if col != "track_name"}
agg_dict["skipped"] = "mean"
merged = merged.groupby("track_name").agg(agg_dict).reset_index()

final = merged[['track_name', 'artist_name', 'album_name_x', 'duration_ms',
                'popularity', 'danceability', 'energy', 'key', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                'tempo', 'mode', 'explicit', 'track_genre', 'skipped']]
final = final.rename(columns={"album_name_x": "album_name",
                              "skipped": "skipped_rate"})

genre_count = final["track_genre"].value_counts()
genre_for_use = sorted(genre_count[genre_count > 100].index)
final = final[final["track_genre"].isin(genre_for_use)].reset_index(drop=True)

encode = one_hot_enc(final, genre_for_use, "track_genre")
for i in range(len(genre_for_use)):
    final[genre_for_use[i]] = encode[i]
train_set, test_set = train_test_split_by_one_cat(final, "track_genre")

stz_feat = ['duration_ms', 'popularity', 'danceability', 'energy', 'key',
            'loudness', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'tempo']
all_feat = stz_feat + ['mode', 'explicit'] + genre_for_use


X_tr_stz = stz(train_set, train_set, stz_feat)[all_feat].to_numpy()
X_tr = np.hstack((np.ones((len(X_tr_stz),1)), X_tr_stz))

Y_tr = train_set["skipped_rate"].to_numpy()
w = np.linalg.inv(X_tr.T @ X_tr) @ X_tr.T @ Y_tr


X_ts_stz = stz(test_set, train_set, stz_feat)[all_feat]
X_ts = np.hstack((np.ones((len(X_ts_stz),1)), X_ts_stz))
pred = w @ X_ts.T
Y_ts = test_set["skipped_rate"]
err = MSE(Y_ts, pred)
print("The mean squared error of the linear regression model for predicting skip rate:",
      err)
