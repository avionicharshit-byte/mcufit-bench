#pragma once

// Models baked into flash, generated from models/*.tflite by
// scripts/gen_model_array.py.

#define MCUFIT_MODEL(sym)                    \
  extern const unsigned char g_##sym[];      \
  extern const unsigned int g_##sym##_len;   \
  extern const char g_##sym##_sha256[];

MCUFIT_MODEL(person_detect)
MCUFIT_MODEL(kws)
MCUFIT_MODEL(ic_resnet)
MCUFIT_MODEL(ad)
