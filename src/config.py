

VERSION = "0.4.0"

DB_CHOICES = ["eleme", "func", "phase", "param", "tdb"]

PHASE_METRICS = {
            "SER": (1,),  # SER is pure
            "BCC": (1, 1),
            "FCC": (1, 3),
            "HCP": (2, 6),
        }

DATA_TYPES = ["dat", "json"]