// c-api-examples/streaming-zipformer-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation

//
// This file demonstrates how to use streaming Zipformer with sherpa-onnx's C and
// with tokens and hotwords loaded from buffered strings instread external files
// API.
// clang-format off
// 
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
// tar xvf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
// rm sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

extern const char* tokens_buf_str;
extern const char* hotwords_buf_str;
int32_t main() {
  const char *wav_filename =
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav";
  const char *encoder_filename =
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/"
      "encoder-epoch-99-avg-1.onnx";
  const char *decoder_filename =
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/"
      "decoder-epoch-99-avg-1.onnx";
  const char *joiner_filename =
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/"
      "joiner-epoch-99-avg-1.onnx";
  const char *provider = "cpu";
  const char *modeling_unit = "bpe";
  const char *bpe_vocab  =  
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/"
      "bpe.vocab";
  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  // Zipformer config
  SherpaOnnxOnlineTransducerModelConfig zipformer_config;
  memset(&zipformer_config, 0, sizeof(zipformer_config));
  zipformer_config.encoder = encoder_filename;
  zipformer_config.decoder = decoder_filename;
  zipformer_config.joiner = joiner_filename;

  // Online model config
  SherpaOnnxOnlineModelConfig online_model_config;
  memset(&online_model_config, 0, sizeof(online_model_config));
  online_model_config.debug = 1;
  online_model_config.num_threads = 1;
  online_model_config.provider = provider;
  online_model_config.tokens_buf_str = tokens_buf_str;
  online_model_config.transducer = zipformer_config;

  // Recognizer config
  SherpaOnnxOnlineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "modified_beam_search";
  recognizer_config.model_config = online_model_config;
  recognizer_config.hotwords_buf_str = hotwords_buf_str;

  SherpaOnnxOnlineRecognizer *recognizer =
      SherpaOnnxCreateOnlineRecognizer(&recognizer_config);

  if (recognizer == NULL) {
    fprintf(stderr, "Please check your config!\n");
    SherpaOnnxFreeWave(wave);
    return -1;
  }

  SherpaOnnxOnlineStream *stream = SherpaOnnxCreateOnlineStream(recognizer);

  const SherpaOnnxDisplay *display = SherpaOnnxCreateDisplay(50);
  int32_t segment_id = 0;

// simulate streaming. You can choose an arbitrary N
#define N 3200

  fprintf(stderr, "sample rate: %d, num samples: %d, duration: %.2f s\n",
          wave->sample_rate, wave->num_samples,
          (float)wave->num_samples / wave->sample_rate);

  int32_t k = 0;
  while (k < wave->num_samples) {
    int32_t start = k;
    int32_t end =
        (start + N > wave->num_samples) ? wave->num_samples : (start + N);
    k += N;

    SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate,
                                         wave->samples + start, end - start);
    while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
      SherpaOnnxDecodeOnlineStream(recognizer, stream);
    }

    const SherpaOnnxOnlineRecognizerResult *r =
        SherpaOnnxGetOnlineStreamResult(recognizer, stream);

    if (strlen(r->text)) {
      SherpaOnnxPrint(display, segment_id, r->text);
    }

    if (SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream)) {
      if (strlen(r->text)) {
        ++segment_id;
      }
      SherpaOnnxOnlineStreamReset(recognizer, stream);
    }

    SherpaOnnxDestroyOnlineRecognizerResult(r);
  }

  // add some tail padding
  float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
  SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail_paddings,
                                       4800);

  SherpaOnnxFreeWave(wave);

  SherpaOnnxOnlineStreamInputFinished(stream);
  while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
    SherpaOnnxDecodeOnlineStream(recognizer, stream);
  }

  const SherpaOnnxOnlineRecognizerResult *r =
      SherpaOnnxGetOnlineStreamResult(recognizer, stream);

  if (strlen(r->text)) {
    SherpaOnnxPrint(display, segment_id, r->text);
  }

  SherpaOnnxDestroyOnlineRecognizerResult(r);

  SherpaOnnxDestroyDisplay(display);
  SherpaOnnxDestroyOnlineStream(stream);
  SherpaOnnxDestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}

const char* hotwords_buf_str = "▁A ▁T ▁P :1.5\n \
▁A ▁B ▁C :3.0";

const char* tokens_buf_str = "<blk> 0 \n \
<sos/eos> 1 \n \
<unk> 2 \n \
S 3 \n \
▁THE 4 \n \
▁A 5 \n \
T 6 \n \
▁AND 7 \n \
ED 8 \n \
▁OF 9 \n \
▁TO 10 \n \
E 11 \n \
D 12 \n \
N 13 \n \
ING 14 \n \
▁IN 15 \n \
Y 16 \n \
M 17 \n \
C 18 \n \
▁I 19 \n \
A 20 \n \
P 21 \n \
▁HE 22 \n \
R 23 \n \
O 24 \n \
L 25 \n \
RE 26 \n \
I 27 \n \
U 28 \n \
ER 29 \n \
▁IT 30 \n \
LY 31 \n \
▁THAT 32 \n \
▁WAS 33 \n \
▁ 34 \n \
▁S 35 \n \
AR 36 \n \
▁BE 37 \n \
F 38 \n \
▁C 39 \n \
IN 40 \n \
B 41 \n \
▁FOR 42 \n \
OR 43 \n \
LE 44 \n \
' 45 \n \
▁HIS 46 \n \
▁YOU 47 \n \
AL 48 \n \
▁RE 49 \n \
V 50 \n \
▁B 51 \n \
G 52 \n \
RI 53 \n \
▁E 54 \n \
▁WITH 55 \n \
▁T 56 \n \
▁AS 57 \n \
LL 58 \n \
▁P 59 \n \
▁HER 60 \n \
ST 61 \n \
▁HAD 62 \n \
▁SO 63 \n \
▁F 64 \n \
W 65 \n \
CE 66 \n \
▁IS 67 \n \
ND 68 \n \
▁NOT 69 \n \
TH 70 \n \
▁BUT 71 \n \
EN 72 \n \
▁SHE 73 \n \
▁ON 74 \n \
VE 75 \n \
ON 76 \n \
SE 77 \n \
▁DE 78 \n \
UR 79 \n \
▁G 80 \n \
CH 81 \n \
K 82 \n \
TER 83 \n \
▁AT 84 \n \
IT 85 \n \
▁ME 86 \n \
RO 87 \n \
NE 88 \n \
RA 89 \n \
ES 90 \n \
IL 91 \n \
NG 92 \n \
IC 93 \n \
▁NO 94 \n \
▁HIM 95 \n \
ENT 96 \n \
IR 97 \n \
▁WE 98 \n \
H 99 \n \
▁DO 100 \n \
▁ALL 101 \n \
▁HAVE 102 \n \
LO 103 \n \
▁BY 104 \n \
▁MY 105 \n \
▁MO 106 \n \
▁THIS 107 \n \
LA 108 \n \
▁ST 109 \n \
▁WHICH 110 \n \
▁CON 111 \n \
▁THEY 112 \n \
CK 113 \n \
TE 114 \n \
▁SAID 115 \n \
▁FROM 116 \n \
▁GO 117 \n \
▁WHO 118 \n \
▁TH 119 \n \
▁OR 120 \n \
▁D 121 \n \
▁W 122 \n \
VER 123 \n \
LI 124 \n \
▁SE 125 \n \
▁ONE 126 \n \
▁CA 127 \n \
▁AN 128 \n \
▁LA 129 \n \
▁WERE 130 \n \
EL 131 \n \
▁HA 132 \n \
▁MAN 133 \n \
▁FA 134 \n \
▁EX 135 \n \
AD 136 \n \
▁SU 137 \n \
RY 138 \n \
▁MI 139 \n \
AT 140 \n \
▁BO 141 \n \
▁WHEN 142 \n \
AN 143 \n \
THER 144 \n \
PP 145 \n \
ATION 146 \n \
▁FI 147 \n \
▁WOULD 148 \n \
▁PRO 149 \n \
OW 150 \n \
ET 151 \n \
▁O 152 \n \
▁THERE 153 \n \
▁HO 154 \n \
ION 155 \n \
▁WHAT 156 \n \
▁FE 157 \n \
▁PA 158 \n \
US 159 \n \
MENT 160 \n \
▁MA 161 \n \
UT 162 \n \
▁OUT 163 \n \
▁THEIR 164 \n \
▁IF 165 \n \
▁LI 166 \n \
▁K 167 \n \
▁WILL 168 \n \
▁ARE 169 \n \
ID 170 \n \
▁RO 171 \n \
DE 172 \n \
TION 173 \n \
▁WA 174 \n \
PE 175 \n \
▁UP 176 \n \
▁SP 177 \n \
▁PO 178 \n \
IGHT 179 \n \
▁UN 180 \n \
RU 181 \n \
▁LO 182 \n \
AS 183 \n \
OL 184 \n \
▁LE 185 \n \
▁BEEN 186 \n \
▁SH 187 \n \
▁RA 188 \n \
▁SEE 189 \n \
KE 190 \n \
UL 191 \n \
TED 192 \n \
▁SA 193 \n \
UN 194 \n \
UND 195 \n \
ANT 196 \n \
▁NE 197 \n \
IS 198 \n \
▁THEM 199 \n \
CI 200 \n \
GE 201 \n \
▁COULD 202 \n \
▁DIS 203 \n \
OM 204 \n \
ISH 205 \n \
HE 206 \n \
EST 207 \n \
▁SOME 208 \n \
ENCE 209 \n \
ITY 210 \n \
IVE 211 \n \
▁US 212 \n \
▁MORE 213 \n \
▁EN 214 \n \
ARD 215 \n \
ATE 216 \n \
▁YOUR 217 \n \
▁INTO 218 \n \
▁KNOW 219 \n \
▁CO 220 \n \
ANCE 221 \n \
▁TIME 222 \n \
▁WI 223 \n \
▁YE 224 \n \
AGE 225 \n \
▁NOW 226 \n \
TI 227 \n \
FF 228 \n \
ABLE 229 \n \
▁VERY 230 \n \
▁LIKE 231 \n \
AM 232 \n \
HI 233 \n \
Z 234 \n \
▁OTHER 235 \n \
▁THAN 236 \n \
▁LITTLE 237 \n \
▁DID 238 \n \
▁LOOK 239 \n \
TY 240 \n \
ERS 241 \n \
▁CAN 242 \n \
▁CHA 243 \n \
▁AR 244 \n \
X 245 \n \
FUL 246 \n \
UGH 247 \n \
▁BA 248 \n \
▁DAY 249 \n \
▁ABOUT 250 \n \
TEN 251 \n \
IM 252 \n \
▁ANY 253 \n \
▁PRE 254 \n \
▁OVER 255 \n \
IES 256 \n \
NESS 257 \n \
ME 258 \n \
BLE 259 \n \
▁M 260 \n \
ROW 261 \n \
▁HAS 262 \n \
▁GREAT 263 \n \
▁VI 264 \n \
TA 265 \n \
▁AFTER 266 \n \
PER 267 \n \
▁AGAIN 268 \n \
HO 269 \n \
SH 270 \n \
▁UPON 271 \n \
▁DI 272 \n \
▁HAND 273 \n \
▁COM 274 \n \
IST 275 \n \
TURE 276 \n \
▁STA 277 \n \
▁THEN 278 \n \
▁SHOULD 279 \n \
▁GA 280 \n \
OUS 281 \n \
OUR 282 \n \
▁WELL 283 \n \
▁ONLY 284 \n \
MAN 285 \n \
▁GOOD 286 \n \
▁TWO 287 \n \
▁MAR 288 \n \
▁SAY 289 \n \
▁HU 290 \n \
TING 291 \n \
▁OUR 292 \n \
RESS 293 \n \
▁DOWN 294 \n \
IOUS 295 \n \
▁BEFORE 296 \n \
▁DA 297 \n \
▁NA 298 \n \
QUI 299 \n \
▁MADE 300 \n \
▁EVERY 301 \n \
▁OLD 302 \n \
▁EVEN 303 \n \
IG 304 \n \
▁COME 305 \n \
▁GRA 306 \n \
▁RI 307 \n \
▁LONG 308 \n \
OT 309 \n \
SIDE 310 \n \
WARD 311 \n \
▁FO 312 \n \
▁WHERE 313 \n \
MO 314 \n \
LESS 315 \n \
▁SC 316 \n \
▁MUST 317 \n \
▁NEVER 318 \n \
▁HOW 319 \n \
▁CAME 320 \n \
▁SUCH 321 \n \
▁RU 322 \n \
▁TAKE 323 \n \
▁WO 324 \n \
▁CAR 325 \n \
UM 326 \n \
AK 327 \n \
▁THINK 328 \n \
▁MUCH 329 \n \
▁MISTER 330 \n \
▁MAY 331 \n \
▁JO 332 \n \
▁WAY 333 \n \
▁COMP 334 \n \
▁THOUGHT 335 \n \
▁STO 336 \n \
▁MEN 337 \n \
▁BACK 338 \n \
▁DON 339 \n \
J 340 \n \
▁LET 341 \n \
▁TRA 342 \n \
▁FIRST 343 \n \
▁JUST 344 \n \
▁VA 345 \n \
▁OWN 346 \n \
▁PLA 347 \n \
▁MAKE 348 \n \
ATED 349 \n \
▁HIMSELF 350 \n \
▁WENT 351 \n \
▁PI 352 \n \
GG 353 \n \
RING 354 \n \
▁DU 355 \n \
▁MIGHT 356 \n \
▁PART 357 \n \
▁GIVE 358 \n \
▁IMP 359 \n \
▁BU 360 \n \
▁PER 361 \n \
▁PLACE 362 \n \
▁HOUSE 363 \n \
▁THROUGH 364 \n \
IAN 365 \n \
▁SW 366 \n \
▁UNDER 367 \n \
QUE 368 \n \
▁AWAY 369 \n \
▁LOVE 370 \n \
QUA 371 \n \
▁LIFE 372 \n \
▁GET 373 \n \
▁WITHOUT 374 \n \
▁PASS 375 \n \
▁TURN 376 \n \
IGN 377 \n \
▁HEAD 378 \n \
▁MOST 379 \n \
▁THOSE 380 \n \
▁SHALL 381 \n \
▁EYES 382 \n \
▁COL 383 \n \
▁STILL 384 \n \
▁NIGHT 385 \n \
▁NOTHING 386 \n \
ITION 387 \n \
HA 388 \n \
▁TELL 389 \n \
▁WORK 390 \n \
▁LAST 391 \n \
▁NEW 392 \n \
▁FACE 393 \n \
▁HI 394 \n \
▁WORD 395 \n \
▁FOUND 396 \n \
▁COUNT 397 \n \
▁OB 398 \n \
▁WHILE 399 \n \
▁SHA 400 \n \
▁MEAN 401 \n \
▁SAW 402 \n \
▁PEOPLE 403 \n \
▁FRIEND 404 \n \
▁THREE 405 \n \
▁ROOM 406 \n \
▁SAME 407 \n \
▁THOUGH 408 \n \
▁RIGHT 409 \n \
▁CHILD 410 \n \
▁FATHER 411 \n \
▁ANOTHER 412 \n \
▁HEART 413 \n \
▁WANT 414 \n \
▁TOOK 415 \n \
OOK 416 \n \
▁LIGHT 417 \n \
▁MISSUS 418 \n \
▁OPEN 419 \n \
▁JU 420 \n \
▁ASKED 421 \n \
PORT 422 \n \
▁LEFT 423 \n \
▁JA 424 \n \
▁WORLD 425 \n \
▁HOME 426 \n \
▁WHY 427 \n \
▁ALWAYS 428 \n \
▁ANSWER 429 \n \
▁SEEMED 430 \n \
▁SOMETHING 431 \n \
▁GIRL 432 \n \
▁BECAUSE 433 \n \
▁NAME 434 \n \
▁TOLD 435 \n \
▁NI 436 \n \
▁HIGH 437 \n \
IZE 438 \n \
▁WOMAN 439 \n \
▁FOLLOW 440 \n \
▁RETURN 441 \n \
▁KNEW 442 \n \
▁EACH 443 \n \
▁KIND 444 \n \
▁JE 445 \n \
▁ACT 446 \n \
▁LU 447 \n \
▁CERTAIN 448 \n \
▁YEARS 449 \n \
▁QUITE 450 \n \
▁APPEAR 451 \n \
▁BETTER 452 \n \
▁HALF 453 \n \
▁PRESENT 454 \n \
▁PRINCE 455 \n \
SHIP 456 \n \
▁ALSO 457 \n \
▁BEGAN 458 \n \
▁HAVING 459 \n \
▁ENOUGH 460 \n \
▁PERSON 461 \n \
▁LADY 462 \n \
▁WHITE 463 \n \
▁COURSE 464 \n \
▁VOICE 465 \n \
▁SPEAK 466 \n \
▁POWER 467 \n \
▁MORNING 468 \n \
▁BETWEEN 469 \n \
▁AMONG 470 \n \
▁KEEP 471 \n \
▁WALK 472 \n \
▁MATTER 473 \n \
▁TEA 474 \n \
▁BELIEVE 475 \n \
▁SMALL 476 \n \
▁TALK 477 \n \
▁FELT 478 \n \
▁HORSE 479 \n \
▁MYSELF 480 \n \
▁SIX 481 \n \
▁HOWEVER 482 \n \
▁FULL 483 \n \
▁HERSELF 484 \n \
▁POINT 485 \n \
▁STOOD 486 \n \
▁HUNDRED 487 \n \
▁ALMOST 488 \n \
▁SINCE 489 \n \
▁LARGE 490 \n \
▁LEAVE 491 \n \
▁PERHAPS 492 \n \
▁DARK 493 \n \
▁SUDDEN 494 \n \
▁REPLIED 495 \n \
▁ANYTHING 496 \n \
▁WONDER 497 \n \
▁UNTIL 498 \n \
Q 499 \n \
#0 500 \n \
#1 501";