| Target    | Size | Time  | Compiler | Opt | Tile Size | Restrict | GOMP                                |      |
| Phenom II | 2048 | 9.52  | gcc      | O2  | 64        |          |                                     |      |
| Phenom II | 2048 | 10.35 | gcc      | O2  | 128       |          |                                     |      |
| Phenom II | 2048 | 6.21  | clang    | O2  | 128       |          |                                     |      |
| Phenom II | 2048 | 5.26  | clang    | O2  | 64        |          |                                     |      |
| Phenom II | 2048 | 5.47  | clang    | O2  | 50        |          |                                     |      |
| Phenom II | 2048 | 4.89  | clang    | O2  | 48        |          |                                     |      |
| Phenom II | 2048 | 4.24  | clang    | O2  | 48        | Yes      |                                     |      |
| Phenom II | 2048 | 17.19 | gcc      | O2  | 48        | Yes      |                                     |      |
| Phenom II | 2048 | 3.50  | clang    | O2  | 64        | Yes      |                                     |      |
| Phenom II | 2048 | 3.47  | clang    | O2  | 192,64    | Yes      |                                     |      |
| Phenom II | 2048 | 5.57  | clang    | O2  | 192,32    | Yes      |                                     |      |
| Phenom II | 2048 | 6.90  | clang    | O2  | 192,96    | Yes      |                                     |      |
| Phenom II | 2048 | 3.56  | clang    | O2  | 128,64    | Yes      |                                     |      |
| Phenom II | 2048 | 1.04  | clang    | O2  | 192,64    | Yes      | parallel for                        |      |
| Phenom II | 2048 | 1.19  | clang    | O2  | 256,64    | Yes      | parallel for                        |      |
| Phenom II | 2048 | 1.02  | clang    | O2  | 128,64    | Yes      | parallel for                        |      |
| Phenom II | 2048 | 0.98  | clang    | O2  | 64,64     | Yes      | parallel for                        |      |
| Phenom II | 4096 | 13.45 | clang    | O2  | 64        | Yes      | parallel for, dynamic               |      |
| Phenom II | 4096 | 10.34 | clang    | O2  | 64        | Yes      | parallel for, static 1              |      |
| Phenom II | 4096 | 10.08 | clang    | O2  | 64        | Yes      | parallel for, static 2              |      |
| Phenom II | 4096 | 9.62  | clang    | O2  | 64        | Yes      | parallel for, static 4              |      |
| Phenom II | 4096 | 9.12  | clang    | O2  | 64        | Yes      | parallel for, static 6              |      |
| Phenom II | 4096 | 12.69 | clang    | O2  | 64        | Yes      | parallel for, static 8              |      |
| Phenom II | 4096 | 15.33 | clang    | O2  | 64,64     | Yes      | parallel for, static 1/1            |      |
| Phenom II | 4096 | 14.97 | clang    | O2  | 64,64     | Yes      | parallel for, dynamic 1/1           |      |
| Phenom II | 4096 | 15.52 | clang    | O2  | 192,64    | Yes      | parallel for, dynamic 1/1           |      |
| Phenom II | 4096 | 14.86 | clang    | O2  | 64        | Yes      | parallel for, dynamic 1             |      |
| Phenom II | 4096 | 11.81 | clang    | O2  | 192       | Yes      | parallel for, static 1              |      |
| Phenom II | 4096 | 11.53 | clang    | O2  | 192       | Yes      | parallel for, static 1, collapse(2) |      |
| Phenom II | 4096 | 10.90 | clang    | O2  | 192       | Yes      | parallel for, static 1, collapse(2) | SIMD |
| Phenom II | 4096 | 16.10 | clang    | O2  | 64        | Yes      | parallel for, static 1, collapse(2) |      |
|           |      |       |          |     |           |          |                                     |      |
| Phenom II | 2048 | 3.51  | clang    | O2  | 256,64    | Yes      |                                     |      |
| Phenom II | 2048 | 4.73  | clang    | O2  | 50        | Yes      |                                     |      |
| Phenom II | 2048 | 6.85  | clang    | O2  | 80        | Yes      |                                     |      |
| Phenom II | 2048 | 5.98  | clang    | O2  | 32        | Yes      |                                     |      |
| Phenom II | 2048 | 6.53  | clang    | O2  | 96        | Yes      |                                     |      |
| Phenom II | 2048 | 4.98  | clang    | O3  | 48        |          |                                     |      |
| Phenom II | 2048 | 10.19 | gcc      | O2  | 48        |          |                                     |      |
| Phenom II | 2048 | 5.22  | clang    | O2  | 44        |          |                                     |      |
| Phenom II | 2048 | 6.19  | clang    | O2  | 32        |          |                                     |      |
| Phenom II | 2048 | 11.59 | clang    | O2  | 16        |          |                                     |      |
|           |      |       |          |     |           |          |                                     |      |
