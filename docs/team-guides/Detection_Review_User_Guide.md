# Detection Review: How to Use

## Open the page

1. Make sure the Backend server, MongoDB/Redis, and the HMI app are running.
2. Go to `http://localhost:3000/admin/detection-review.html`, or click **Detection Review** in the sidebar under **Echo**.

## Steps

1. Click a detection on the left list.
2. On the right, you will see:
   - The detection's own audio and confidence score.
   - A status badge.
   - The 5 most similar past detections, with audio for each.
3. Listen and compare by ear before trusting the result.

## Badge meanings

| Badge | Meaning |
|---|---|
| **Consistent match** (green) | The similar detections agree on species. Normal case. |
| **Ambiguous** (yellow) | The similar detections do not agree on species. Double check before trusting the label. |
| **Novel** (red) | Nothing similar was found before. Worth a manual listen. |

## Reading the similarity list

- Each row shows a species, its audio, and a similarity score (0 to 1, higher = more similar).
- Rows with similarity below 0.6 are shown faded with the label "weak, not a strong precedent". These are not real matches, only the closest thing available. Ignore them when judging the detection.

## If the page does not load

The Backend server is probably not running (it is started manually, separate from Docker). Start it and reload the page.
