# EnvatoDownloader Usage Guide

Here is a quick summary of the available parameters for `EnvatoDownloader.py`:

| Parameter | Required | Description | Example |
| :--- | :--- | :--- | :--- |
| `--term` | **Yes** | The exact search phrase you want to query on Envato Elements. | `--term "Modular building construction"` |
| `--segment` | **Yes** | A unique ID or segment number used to name the isolated downloads folder. | `--segment "seg_001"` |
| `--count` | No | Overrides the default number of videos to download. Defaults to `5`. | `--count 10` |

### Example Usage:
```bash
python3 EnvatoDownloader.py --term "Buildings are assembled in a modular manner" --segment "seg_001" --count 5
```
