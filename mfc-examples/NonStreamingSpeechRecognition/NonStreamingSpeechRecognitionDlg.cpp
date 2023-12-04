
// NonStreamingSpeechRecognitionDlg.cpp : implementation file
//

// clang-format off
#include "pch.h"
#include "framework.h"
#include "afxdialogex.h"
#include "NonStreamingSpeechRecognition.h"
#include "NonStreamingSpeechRecognitionDlg.h"
// clang-format on

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

Microphone::Microphone() {
  PaError err = Pa_Initialize();
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(-2);
  }
}

Microphone::~Microphone() {
  PaError err = Pa_Terminate();
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(-2);
  }
}

// see
// https://stackoverflow.com/questions/7153935/how-to-convert-utf-8-stdstring-to-utf-16-stdwstring
static std::wstring Utf8ToUtf16(const std::string &utf8) {
  std::vector<unsigned long> unicode;
  size_t i = 0;
  while (i < utf8.size()) {
    unsigned long uni;
    size_t todo;
    bool error = false;
    unsigned char ch = utf8[i++];
    if (ch <= 0x7F) {
      uni = ch;
      todo = 0;
    } else if (ch <= 0xBF) {
      throw std::logic_error("not a UTF-8 string");
    } else if (ch <= 0xDF) {
      uni = ch & 0x1F;
      todo = 1;
    } else if (ch <= 0xEF) {
      uni = ch & 0x0F;
      todo = 2;
    } else if (ch <= 0xF7) {
      uni = ch & 0x07;
      todo = 3;
    } else {
      throw std::logic_error("not a UTF-8 string");
    }
    for (size_t j = 0; j < todo; ++j) {
      if (i == utf8.size()) throw std::logic_error("not a UTF-8 string");
      unsigned char ch = utf8[i++];
      if (ch < 0x80 || ch > 0xBF) throw std::logic_error("not a UTF-8 string");
      uni <<= 6;
      uni += ch & 0x3F;
    }
    if (uni >= 0xD800 && uni <= 0xDFFF)
      throw std::logic_error("not a UTF-8 string");
    if (uni > 0x10FFFF) throw std::logic_error("not a UTF-8 string");
    unicode.push_back(uni);
  }
  std::wstring utf16;
  for (size_t i = 0; i < unicode.size(); ++i) {
    unsigned long uni = unicode[i];
    if (uni <= 0xFFFF) {
      utf16 += (wchar_t)uni;
    } else {
      uni -= 0x10000;
      utf16 += (wchar_t)((uni >> 10) + 0xD800);
      utf16 += (wchar_t)((uni & 0x3FF) + 0xDC00);
    }
  }
  return utf16;
}

static std::string Cat(const std::vector<std::string> &results) {
  std::ostringstream os;
  std::string sep;

  int i = 0;
  for (i = 0; i != results.size(); ++i) {
    os << sep << i << ": " << results[i];
    sep = "\r\n";
  }

  return os.str();
}

// CNonStreamingSpeechRecognitionDlg dialog

CNonStreamingSpeechRecognitionDlg::CNonStreamingSpeechRecognitionDlg(
    CWnd *pParent /*=nullptr*/)
    : CDialogEx(IDD_NONSTREAMINGSPEECHRECOGNITION_DIALOG, pParent) {
  m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

CNonStreamingSpeechRecognitionDlg::~CNonStreamingSpeechRecognitionDlg() {
  if (recognizer_) {
    DestroyOfflineRecognizer(recognizer_);
    recognizer_ = nullptr;
  }
}

void CNonStreamingSpeechRecognitionDlg::DoDataExchange(CDataExchange *pDX) {
  CDialogEx::DoDataExchange(pDX);
  DDX_Control(pDX, IDC_EDIT1, my_text_);
  DDX_Control(pDX, IDOK, my_btn_);
}

BEGIN_MESSAGE_MAP(CNonStreamingSpeechRecognitionDlg, CDialogEx)
ON_WM_PAINT()
ON_WM_QUERYDRAGICON()
ON_BN_CLICKED(IDOK, &CNonStreamingSpeechRecognitionDlg::OnBnClickedOk)
END_MESSAGE_MAP()

// CNonStreamingSpeechRecognitionDlg message handlers

BOOL CNonStreamingSpeechRecognitionDlg::OnInitDialog() {
  CDialogEx::OnInitDialog();

  // Set the icon for this dialog.  The framework does this automatically
  //  when the application's main window is not a dialog
  SetIcon(m_hIcon, TRUE);   // Set big icon
  SetIcon(m_hIcon, FALSE);  // Set small icon

  // TODO: Add extra initialization here
  InitMicrophone();

  return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CNonStreamingSpeechRecognitionDlg::OnPaint() {
  if (IsIconic()) {
    CPaintDC dc(this);  // device context for painting

    SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()),
                0);

    // Center icon in client rectangle
    int cxIcon = GetSystemMetrics(SM_CXICON);
    int cyIcon = GetSystemMetrics(SM_CYICON);
    CRect rect;
    GetClientRect(&rect);
    int x = (rect.Width() - cxIcon + 1) / 2;
    int y = (rect.Height() - cyIcon + 1) / 2;

    // Draw the icon
    dc.DrawIcon(x, y, m_hIcon);
  } else {
    CDialogEx::OnPaint();
  }
}

// The system calls this function to obtain the cursor to display while the user
// drags
//  the minimized window.
HCURSOR CNonStreamingSpeechRecognitionDlg::OnQueryDragIcon() {
  return static_cast<HCURSOR>(m_hIcon);
}

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void *user_data) {
  auto dlg = reinterpret_cast<CNonStreamingSpeechRecognitionDlg *>(user_data);
  auto begin = reinterpret_cast<const float *>(input_buffer);
  auto end = begin + frames_per_buffer;
  dlg->samples_.insert(dlg->samples_.end(), begin, end);

  return dlg->started_ ? paContinue : paComplete;
}

void CNonStreamingSpeechRecognitionDlg::OnBnClickedOk() {
  if (!recognizer_) {
    AppendLineToMultilineEditCtrl("Creating recognizer...");
    AppendLineToMultilineEditCtrl("It will take several seconds. Please wait");
    InitRecognizer();
    if (!recognizer_) {
      // failed to create the recognizer
      return;
    }
    AppendLineToMultilineEditCtrl("Recognizer created!");
  }

  if (!started_) {
    samples_.clear();
    started_ = true;

    PaStreamParameters param;
    param.device = Pa_GetDefaultInputDevice();
    const PaDeviceInfo *info = Pa_GetDeviceInfo(param.device);
    param.channelCount = 1;
    param.sampleFormat = paFloat32;
    param.suggestedLatency = info->defaultLowInputLatency;
    param.hostApiSpecificStreamInfo = nullptr;
    float sample_rate = static_cast<float>(config_.feat_config.sample_rate);
    pa_stream_ = nullptr;
    PaError err =
        Pa_OpenStream(&pa_stream_, &param, nullptr, /* &outputParameters, */
                      sample_rate,
                      0,          // frames per buffer
                      paClipOff,  // we won't output out of range samples
                                  // so don't bother clipping them
                      RecordCallback, this);
    if (err != paNoError) {
      AppendLineToMultilineEditCtrl(std::string("PortAudio error: ") +
                                    Pa_GetErrorText(err));
      my_btn_.EnableWindow(FALSE);
      return;
    }

    err = Pa_StartStream(pa_stream_);
    if (err != paNoError) {
      AppendLineToMultilineEditCtrl(std::string("PortAudio error: ") +
                                    Pa_GetErrorText(err));
      my_btn_.EnableWindow(FALSE);
      return;
    }
    AppendLineToMultilineEditCtrl(
        "\r\nStarted! Please speak and click stop.\r\n");
    my_btn_.SetWindowText(_T("Stop"));

  } else {
    started_ = false;

    Pa_Sleep(200);  // sleep for 200ms
    if (pa_stream_) {
      PaError err = Pa_CloseStream(pa_stream_);
      if (err != paNoError) {
        AppendLineToMultilineEditCtrl(std::string("PortAudio error: ") +
                                      Pa_GetErrorText(err));
        my_btn_.EnableWindow(FALSE);
        return;
      }
    }
    pa_stream_ = nullptr;

    SherpaOnnxOfflineStream *stream = CreateOfflineStream(recognizer_);

    AcceptWaveformOffline(stream, config_.feat_config.sample_rate,
                          samples_.data(), static_cast<int32_t>(samples_.size()));
    DecodeOfflineStream(recognizer_, stream);
    auto r = GetOfflineStreamResult(stream);
    results_.emplace_back(r->text);

    auto str = Utf8ToUtf16(Cat(results_).c_str());
    my_text_.SetWindowText(str.c_str());
    my_text_.SetFocus();
    my_text_.SetSel(-1);

    DestroyOfflineRecognizerResult(r);

    DestroyOfflineStream(stream);
    // AfxMessageBox("Stopped", MB_OK);
    my_btn_.SetWindowText(_T("Start"));
    AppendLineToMultilineEditCtrl("\r\nStopped. Please click start and speak");
  }
}

void CNonStreamingSpeechRecognitionDlg::InitMicrophone() {
  int default_device = Pa_GetDefaultInputDevice();
  int device_count = Pa_GetDeviceCount();
  if (default_device == paNoDevice) {
    // CString str;
    // str.Format(_T("No default input device found!"));
    // AfxMessageBox(str, MB_OK | MB_ICONSTOP);
    // exit(-1);
    AppendLineToMultilineEditCtrl("No default input device found!");
    my_btn_.EnableWindow(FALSE);
    return;
  }
  AppendLineToMultilineEditCtrl(std::string("Selected device ") +
                                Pa_GetDeviceInfo(default_device)->name);
}

bool CNonStreamingSpeechRecognitionDlg::Exists(const std::string &filename) {
  std::ifstream is(filename);
  return is.good();
}

void CNonStreamingSpeechRecognitionDlg::ShowInitRecognizerHelpMessage() {
  my_btn_.EnableWindow(FALSE);
  std::string msg =
      "\r\nPlease go to\r\n"
      "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html "
      "\r\n";
  msg += "to download a non-streaming model, i.e., an offline model.\r\n";
  msg += "You need to rename them after downloading\r\n\r\n";
  msg += "It supports transducer, paraformer, and whisper models.\r\n\r\n";
  msg +=
      "We give three examples below to show you how to download models\r\n\r\n";
  msg += "(1) Transducer\r\n\r\n";
  msg +=
      "We use "
      "https://huggingface.co/pkufool/"
      "icefall-asr-zipformer-wenetspeech-20230615 below\r\n";
  msg +=
      "wget "
      "https://huggingface.co/pkufool/"
      "icefall-asr-zipformer-wenetspeech-20230615/resolve/main/exp/"
      "encoder-epoch-12-avg-4.onnx\r\n";
  msg +=
      "wget "
      "https://huggingface.co/pkufool/"
      "icefall-asr-zipformer-wenetspeech-20230615/resolve/main/exp/"
      "decoder-epoch-12-avg-4.onnx\r\n";
  msg +=
      "wget "
      "https://huggingface.co/pkufool/"
      "icefall-asr-zipformer-wenetspeech-20230615/resolve/main/exp/"
      "joiner-epoch-12-avg-4.onnx\r\n";
  msg += "\r\n Now rename them\r\n";
  msg += "mv encoder-epoch-12-avg-4.onnx encoder.onnx\r\n";
  msg += "mv decoder-epoch-12-avg-4.onnx decoder.onnx\r\n";
  msg += "mv joiner-epoch-12-avg-4.onnx joiner.onnx\r\n\r\n";
  msg += "(2) Paraformer\r\n\r\n";
  msg +=
      "wget "
      "https://huggingface.co/csukuangfj/"
      "sherpa-onnx-paraformer-zh-2023-03-28/resolve/main/model.onnx\r\n";
  msg +=
      "wget "
      "https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28/"
      "resolve/main/tokens.txt\r\n\r\n";
  msg += "\r\n Now rename them\r\n";
  msg += "mv model.onnx paraformer.onnx\r\n\r\n";
  msg += "(3) Whisper\r\n\r\n";
  msg +=
      "wget "
      "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/"
      "main/tiny.en-encoder.onnx\r\n";
  msg +=
      "wget "
      "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/"
      "main/tiny.en-decoder.onnx\r\n";
  msg +=
      "wget "
      "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/"
      "main/tiny.en-tokens.txt\r\n";
  msg += "\r\n Now rename them\r\n";
  msg += "mv tiny.en-encoder.onnx whisper-encoder.onnx\r\n";
  msg += "mv tiny.en-decoder.onnx whisper-decoder.onnx\r\n";
  msg += "\r\n";
  msg += "That's it!\r\n";

  AppendLineToMultilineEditCtrl(msg);
}

void CNonStreamingSpeechRecognitionDlg::InitWhisper() {
  std::string whisper_encoder = "./whisper-encoder.onnx";
  std::string whisper_decoder = "./whisper-decoder.onnx";

  std::string tokens = "./tokens.txt";

  bool is_ok = true;

  if (Exists("./whisper-encoder.int8.onnx")) {
    whisper_encoder = "./whisper-encoder.int8.onnx";
  } else if (!Exists(whisper_encoder)) {
    std::string msg = whisper_encoder + " does not exist!";
    AppendLineToMultilineEditCtrl(msg);
    is_ok = false;
  }

  if (Exists("./whisper-decoder.int8.onnx")) {
    whisper_decoder = "./whisper-decoder.int8.onnx";
  } else if (!Exists(whisper_decoder)) {
    std::string msg = whisper_decoder + " does not exist!";
    AppendLineToMultilineEditCtrl(msg);
    is_ok = false;
  }

  if (!Exists(tokens)) {
    std::string msg = tokens + " does not exist!";
    AppendLineToMultilineEditCtrl(msg);
    is_ok = false;
  }

  if (!is_ok) {
    ShowInitRecognizerHelpMessage();
    return;
  }

  memset(&config_, 0, sizeof(config_));

  config_.feat_config.sample_rate = 16000;
  config_.feat_config.feature_dim = 80;

  config_.model_config.whisper.encoder = whisper_encoder.c_str();
  config_.model_config.whisper.decoder = whisper_decoder.c_str();
  config_.model_config.tokens = tokens.c_str();
  config_.model_config.num_threads = 1;
  config_.model_config.debug = 1;
  config_.model_config.model_type = "whisper";

  config_.decoding_method = "greedy_search";
  config_.max_active_paths = 4;

  recognizer_ = CreateOfflineRecognizer(&config_);
}

void CNonStreamingSpeechRecognitionDlg::InitParaformer() {
  std::string paraformer = "./paraformer.onnx";
  std::string tokens = "./tokens.txt";

  bool is_ok = true;

  if (Exists("./paraformer.int8.onnx")) {
    paraformer = "./paraformer.int8.onnx";
  } else if (!Exists(paraformer)) {
    std::string msg = paraformer + " does not exist!";
    AppendLineToMultilineEditCtrl(msg);
    is_ok = false;
  }

  if (!Exists(tokens)) {
    std::string msg = tokens + " does not exist!";
    AppendLineToMultilineEditCtrl(msg);
    is_ok = false;
  }

  if (!is_ok) {
    ShowInitRecognizerHelpMessage();
    return;
  }

  memset(&config_, 0, sizeof(config_));

  config_.feat_config.sample_rate = 16000;
  config_.feat_config.feature_dim = 80;

  config_.model_config.paraformer.model = paraformer.c_str();
  config_.model_config.tokens = tokens.c_str();
  config_.model_config.num_threads = 1;
  config_.model_config.debug = 1;
  config_.model_config.model_type = "paraformer";

  config_.decoding_method = "greedy_search";
  config_.max_active_paths = 4;

  recognizer_ = CreateOfflineRecognizer(&config_);
}

void CNonStreamingSpeechRecognitionDlg::InitRecognizer() {
  if (Exists("./paraformer.onnx") || Exists("./paraformer.int8.onnx")) {
    InitParaformer();
    return;
  }

  if (Exists("./whisper-encoder.onnx") || Exists("./whisper-encoder.int8.onnx")) {
    InitWhisper();
    return;
  }

  // assume it is transducer

  std::string encoder = "./encoder.onnx";
  std::string decoder = "./decoder.onnx";
  std::string joiner = "./joiner.onnx";
  std::string tokens = "./tokens.txt";

  bool is_ok = true;
  if (!Exists(encoder)) {
    std::string msg = encoder + " does not exist!";
    AppendLineToMultilineEditCtrl(msg);
    is_ok = false;
  }

  if (!Exists(decoder)) {
    std::string msg = decoder + " does not exist!";
    AppendLineToMultilineEditCtrl(msg);
    is_ok = false;
  }

  if (!Exists(joiner)) {
    std::string msg = joiner + " does not exist!";
    AppendLineToMultilineEditCtrl(msg);
    is_ok = false;
  }

  if (!Exists(tokens)) {
    std::string msg = tokens + " does not exist!";
    AppendLineToMultilineEditCtrl(msg);
    is_ok = false;
  }

  if (!is_ok) {
    ShowInitRecognizerHelpMessage();
    return;
  }
  memset(&config_, 0, sizeof(config_));

  config_.feat_config.sample_rate = 16000;
  config_.feat_config.feature_dim = 80;

  config_.model_config.transducer.encoder = encoder.c_str();
  config_.model_config.transducer.decoder = decoder.c_str();
  config_.model_config.transducer.joiner = joiner.c_str();
  config_.model_config.tokens = tokens.c_str();
  config_.model_config.num_threads = 1;
  config_.model_config.debug = 0;
  config_.model_config.model_type = "transducer";

  config_.decoding_method = "greedy_search";
  config_.max_active_paths = 4;

  recognizer_ = CreateOfflineRecognizer(&config_);
}

void CNonStreamingSpeechRecognitionDlg::AppendTextToEditCtrl(
    const std::string &s) {
  // get the initial text length
  int nLength = my_text_.GetWindowTextLength();
  // put the selection at the end of text
  my_text_.SetSel(nLength, nLength);
  // replace the selection

  std::wstring wstr = Utf8ToUtf16(s);

  my_text_.ReplaceSel(wstr.c_str());
}

void CNonStreamingSpeechRecognitionDlg::AppendLineToMultilineEditCtrl(
    const std::string &s) {
  AppendTextToEditCtrl("\r\n" + s);
}
