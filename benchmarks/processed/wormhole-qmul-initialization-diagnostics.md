# Wormhole qmul initialization diagnostics

The elevated first-case H2D and warmup times follow the first device submission regardless of whether the first size is 4,096, 65,536, or 262,144. Program construction is not the dominant change, and the preregistered stability order is unchanged.

| order | occurrence | N | allocation ms | build ms | H2D ms | prewarm ms | warmup ms | median ms | D2H ms | cleanup ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 4,096 | 0.226 | 0.555 | 14.236 | 0.100 | 16.432 | 1.795 | 0.253 | 0.032 |
| A | 2 | 65,536 | 0.025 | 0.498 | 0.301 | 0.046 | 1.257 | 2.155 | 1.630 | 0.016 |
| A | 3 | 262,144 | 0.066 | 0.628 | 0.765 | 0.042 | 1.743 | 4.298 | 6.213 | 0.031 |
| A | 4 | 4,096 | 0.040 | 0.241 | 0.103 | 0.039 | 0.910 | 1.502 | 0.167 | 0.011 |
| B | 1 | 262,144 | 0.245 | 0.923 | 12.067 | 0.100 | 17.433 | 4.229 | 5.693 | 0.047 |
| B | 2 | 65,536 | 0.046 | 0.574 | 0.258 | 0.039 | 1.253 | 1.903 | 1.114 | 0.011 |
| B | 3 | 4,096 | 0.017 | 0.142 | 0.067 | 0.033 | 0.789 | 1.323 | 0.079 | 0.010 |
| C | 1 | 65,536 | 0.230 | 0.872 | 8.748 | 0.094 | 16.309 | 2.064 | 1.714 | 0.033 |
| C | 2 | 65,536 | 0.027 | 0.507 | 0.214 | 0.038 | 1.545 | 2.055 | 1.413 | 0.016 |
