import taglib

song = taglib.File('../Experiments/0_InitialProcessing/combined/1_1710389871324.wav')

song.tags["ARTIST"] = ["Artist Name"]
song.tags["TITLE"] = ["Track Title"]
song.tags["COMMENT"] = ["Your comment here"]

song.save()

# Re-check the updated tags
print("Updated Tags:", song.tags)

# Close the file
song.close()