
// NonStreamingTextToSpeechDlg.cpp : implementation file
//

#include "pch.h"
#include "framework.h"
#include "NonStreamingTextToSpeech.h"
#include "NonStreamingTextToSpeechDlg.h"
#include "afxdialogex.h"

#include <fstream>
#include <mutex>  // NOLINT
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT
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

// NOTE(fangjun): Code is copied from
// https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/sherpa-onnx-offline-tts-play.cc#L22
static std::condition_variable g_cv;
static std::mutex g_cv_m;

struct Samples {
  std::vector<float> data;
  int32_t consumed = 0;
};

struct Buffer {
  std::queue<Samples> samples;
  std::mutex mutex;
};

static Buffer g_buffer;

static bool g_started = false;
static bool g_stopped = false;
static bool g_killed = false;

static int32_t AudioGeneratedCallback(const float *s, int32_t n) {
  if (n > 0) {
    Samples samples;
    samples.data = std::vector<float>{s, s + n};

    std::lock_guard<std::mutex> lock(g_buffer.mutex);
    g_buffer.samples.push(std::move(samples));
    g_started = true;
  }
  if (g_killed) {
    return 0;
  }
  return 1;
}

static int PlayCallback(const void * /*in*/, void *out,
                        unsigned long _n,  // NOLINT
                        const PaStreamCallbackTimeInfo * /*time_info*/,
                        PaStreamCallbackFlags /*status_flags*/,
                        void * /*user_data*/) {
  int32_t n = static_cast<int32_t>(_n);
  if (g_killed) {
    return paComplete;
  }

  float *pout = reinterpret_cast<float *>(out);
  std::lock_guard<std::mutex> lock(g_buffer.mutex);

  if (g_buffer.samples.empty()) {
    if (g_stopped) {
      // no more data is available and we have processed all of the samples
      return paComplete;
    }

    // The current sentence is so long, though very unlikely, that
    // the model has not finished processing it yet.
    std::fill_n(pout, n, 0);

    return paContinue;
  }

  int32_t k = 0;
  for (; k < n && !g_buffer.samples.empty();) {
    int32_t this_block = n - k;

    auto &p = g_buffer.samples.front();

    int32_t remaining = static_cast<int32_t>(p.data.size()) - p.consumed;

    if (this_block <= remaining) {
      std::copy(p.data.begin() + p.consumed,
                p.data.begin() + p.consumed + this_block, pout + k);
      p.consumed += this_block;

      k = n;

      if (p.consumed == p.data.size()) {
        g_buffer.samples.pop();
      }
      break;
    }

    std::copy(p.data.begin() + p.consumed, p.data.end(), pout + k);
    k += static_cast<int32_t>(p.data.size()) - p.consumed;
    g_buffer.samples.pop();
  }

  if (k < n) {
    std::fill_n(pout + k, n - k, 0);
  }

  if (g_stopped && g_buffer.samples.empty()) {
    return paComplete;
  }

  return paContinue;
}

static void PlayCallbackFinished(void *userData) { g_cv.notify_all(); }

static void StartPlayback(int32_t sample_rate) {
  int32_t frames_per_buffer = 1024;
  PaStreamParameters outputParameters;
  PaStream *stream;
  PaError err;

  outputParameters.device =
      Pa_GetDefaultOutputDevice(); /* default output device */

  outputParameters.channelCount = 1;         /* stereo output */
  outputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */
  outputParameters.suggestedLatency =
      Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
  outputParameters.hostApiSpecificStreamInfo = nullptr;

  err = Pa_OpenStream(&stream, nullptr, /* no input */
                      &outputParameters, sample_rate, frames_per_buffer,
                      paClipOff,  // we won't output out of range samples so
                                  //   don't bother clipping them
                      PlayCallback, nullptr);
  if (err != paNoError) {
    fprintf(stderr, "%d portaudio error: %s\n", __LINE__, Pa_GetErrorText(err));
    return;
  }

  err = Pa_SetStreamFinishedCallback(stream, &PlayCallbackFinished);
  if (err != paNoError) {
    fprintf(stderr, "%d portaudio error: %s\n", __LINE__, Pa_GetErrorText(err));
    return;
  }

  err = Pa_StartStream(stream);
  if (err != paNoError) {
    fprintf(stderr, "%d portaudio error: %s\n", __LINE__, Pa_GetErrorText(err));
    return;
  }

  std::unique_lock<std::mutex> lock(g_cv_m);
  while (!g_killed && !g_stopped &&
         (!g_started || (g_started && !g_buffer.samples.empty()))) {
    g_cv.wait(lock);
  }

  err = Pa_StopStream(stream);
  if (err != paNoError) {
    return;
  }

  err = Pa_CloseStream(stream);
  if (err != paNoError) {
    return;
  }
}


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CNonStreamingTextToSpeechDlg dialog

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

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CNonStreamingTextToSpeechDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void AppendTextToEditCtrl(CEdit& e, const std::string &s) {
  // get the initial text length
  int nLength = e.GetWindowTextLength();
  // put the selection at the end of text
  e.SetSel(nLength, nLength);
  // replace the selection

  std::wstring wstr = Utf8ToUtf16(s);

  // my_text_.ReplaceSel(wstr.c_str());
  e.ReplaceSel(wstr.c_str());
}

void AppendLineToMultilineEditCtrl(CEdit& e, const std::string &s) {
  AppendTextToEditCtrl(e, "\r\n" + s);
}


CNonStreamingTextToSpeechDlg::CNonStreamingTextToSpeechDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_NONSTREAMINGTEXTTOSPEECH_DIALOG, pParent)
       {
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CNonStreamingTextToSpeechDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialogEx::DoDataExchange(pDX);
        DDX_Control(pDX, IDC_HINT, my_hint_);
        DDX_Control(pDX, IDC_SPEAKER, speaker_id_);
        DDX_Control(pDX, IDC_SPEED, speed_);
        DDX_Control(pDX, IDOK, generate_btn_);
        DDX_Control(pDX, IDC_TEXT, my_text_);
        DDX_Control(pDX, IDC_OUTPUT_FILENAME, output_filename_);
}

BEGIN_MESSAGE_MAP(CNonStreamingTextToSpeechDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
        ON_BN_CLICKED(IDOK, &CNonStreamingTextToSpeechDlg::OnBnClickedOk)
        ON_BN_CLICKED(IDC_STOP, &CNonStreamingTextToSpeechDlg::OnBnClickedStop)
        END_MESSAGE_MAP()


// CNonStreamingTextToSpeechDlg message handlers

BOOL CNonStreamingTextToSpeechDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
    Init();

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CNonStreamingTextToSpeechDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon            .  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CNonStreamingTextToSpeechDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon =             GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

bool Exists(const std::string &filename) {
  std::ifstream is(filename);
  return is.good();
}

void CNonStreamingTextToSpeechDlg::InitHint() {
    AppendLineToMultilineEditCtrl(my_hint_, "Speaker ID: Used only for multi-speaker models. Example value: 0");
    AppendLineToMultilineEditCtrl(my_hint_, "Speed: Larger -> Faster in speech speed. Example value: 1.0");
    AppendLineToMultilineEditCtrl(my_hint_, "\r\nPlease input your text and click the button Generate");

}

void CNonStreamingTextToSpeechDlg::Init() {
    InitHint();
    speaker_id_.SetWindowText(Utf8ToUtf16("0").c_str());
    speed_.SetWindowText(Utf8ToUtf16("1.0").c_str());
    output_filename_.SetWindowText(Utf8ToUtf16("./generated.wav").c_str());

	bool ok = true;
  std::string error_message = "--------------------\r\n";
  if (!Exists("./model.onnx")) {
    error_message += "Cannot find ./model.onnx\r\n";
    ok = false;
  }

  if (!Exists("./tokens.txt")) {
    error_message += "Cannot find ./tokens.txt\r\n";
    ok = false;
  }
  // it is OK to leave lexicon.txt and espeak-ng-data empty
  // since models using characters don't need them

  if (!ok) {
    generate_btn_.EnableWindow(FALSE);
    error_message +=
        "\r\nPlease refer to\r\n"
        "https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models"
        "\r\nor\r\n"
        "https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models";

    error_message += "\r\nto download models.\r\n";
    error_message += "\r\nWe give several examples below\r\n";
    error_message += "      1. Use a Kokoro TTS model (multi-lingual, e.g, English + Chinese)\r\n";
    error_message += "      2. Use a Kokoro TTS model (English only)\r\n";
    error_message += "      3. Use a VITS Piper TTS model\r\n";
    error_message += "      4. Use a VITS Chinese TTS model\r\n";
    error_message += "      5. Use a Matcha TTS model\r\n";
    error_message += "\r\n";

    error_message += 
        "----------1. Use a Kokoro TTS model (multi-lingual, eg., English + Chinese)----------\r\n"
        "(a) Download the model from \r\n"
        "     https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2\r\n"
        "(b) Uncompress it and you will get a directory kokoro-multi-lang-v1_0\r\n"
        "(c) Switch to the directory kokoro-multi-lang-v1_0\r\n"
        "(d) Copy the current exe to the directory kokoro-multi-lang-v1_0\r\n"
        "(e).Done! You can now run the exe in the directory kokoro-multi-lang-v1_0\r\n";

    error_message +=  "\r\n";

    error_message += 
        "----------2. Use a Kokoro TTS model (English only)----------\r\n"
        "(a) Download the model from \r\n"
        "     https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2\r\n"
        "(b) Uncompress it and you will get a directory kokoro-en-v0_19\r\n"
        "(c) Switch to the directory kokoro-en-v0_19\r\n"
        "(d) Copy the current exe to the directory kokoro-en-v0_19\r\n"
        "(e).Done! You can now run the exe in the directory kokoro-en-v0_19\r\n";

    error_message +=  "\r\n";

    error_message += 
        "----------3. Use a VITS Piper TTS model----------\r\n"
        "(a) Download the model from \r\n"
        "     https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2\r\n"
        "(b) Uncompress it and you will get a directory vits-piper-en_US-amy-low\r\n"
        "(c) Switch to the directory vits-piper-en_US-amy-low \r\n"
        "(d) Rename en_US-amy-low.onnx to model.onnx\r\n"
        "(e) Copy the current exe to the directory vits-piper-en_US-amy-low\r\n"
        "(f) Done! You can now run the exe in the directory vits-piper-en_US-amy-low\r\n";

    error_message +=  "\r\n";

    error_message += 
        "----------4. Use a VITS Chinese TTS model----------\r\n"
        "(a) Download the model from \r\n"
        "     https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2\r\n"
        "(b) Uncompress it and you will get a directory sherpa-onnx-vits-zh-ll\r\n"
        "(c) Switch to the directory sherpa-onnx-vits-zh-ll\r\n"
        "(d) Copy the current exe to the directory sherpa-onnx-vits-zh-ll\r\n"
        "(e) Done! You can now run the exe in the directory sherpa-onnx-vits-zh-ll\r\n";

    error_message +=  "\r\n";

    error_message += 
        "----------5. Use a Matcha TTS model----------\r\n"
        "(a) Download the model from \r\n"
        "     https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2\r\n"
        "(b) Uncompress it and you will get a directory matcha-icefall-zh-baker\r\n"
        "(c) Switch to the directory matcha-icefall-zh-baker\r\n"
        "(d) Rename model-steps-3.onnx to model.onnx\r\n"
        "(e) Download a vocoder model from \r\n"
        "      https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx\r\n"
        "(f) Rename vocos-22khz-univ.onnx to vocos.onnx\r\n"
        "(g) Remember to put vocos.onnx in the directory matcha-icefall-zh-baker\r\n"
        "(h) Copy the current exe to the directory matcha-icefall-zh-baker\r\n"
        "(i) Done! You can now run the exe in the directory matcha-icefall-zh-baker\r\n";

    AppendLineToMultilineEditCtrl(my_hint_, error_message);
    return;
  }

  // Now init tts
  SherpaOnnxOfflineTtsConfig config;
  memset(&config, 0, sizeof(config));
  config.model.debug = 0;
  config.model.num_threads = 4;
  config.model.provider = "cpu";

  if (Exists("./voices.bin")) {
    // it is a kokoro tts model
    config.model.kokoro.model = "./model.onnx";
    config.model.kokoro.voices = "./voices.bin";
    config.model.kokoro.tokens = "./tokens.txt";
    config.model.kokoro.data_dir = "./espeak-ng-data";
    if (Exists("./dict/jieba.dict.utf8") && Exists("./lexicon-zh.txt")) {
      config.model.kokoro.dict_dir = "./dict";
      config.model.kokoro.lexicon = "./lexicon-us-en.txt,./lexicon-zh.txt";
    }
  } else if (Exists("./hifigan.onnx") || Exists("./vocos.onnx")) {
    // it is a matcha tts model
    config.model.matcha.acoustic_model = "./model.onnx";

    if (Exists("./hifigan.onnx")) {
      config.model.matcha.vocoder = "./hifigan.onnx";
    } else if (Exists("./vocos.onnx")) {
      config.model.matcha.vocoder = "./vocos.onnx";
    }

    config.model.matcha.tokens = "./tokens.txt";

    if (Exists("./espeak-ng-data/phontab")) {
      config.model.matcha.data_dir = "./espeak-ng-data";
    }

    if(Exists("./lexicon.txt")) {
      config.model.matcha.lexicon = "./lexicon.txt";
    }

    if (Exists("./dict/jieba.dict.utf8")) {
      config.model.matcha.dict_dir = "./dict";
    }
  } else {
    // it is a vits tts model
    config.model.vits.model = "./model.onnx";
    config.model.vits.tokens = "./tokens.txt";
    if (Exists("./espeak-ng-data/phontab")) {
      config.model.vits.data_dir = "./espeak-ng-data";
    } 

    if (Exists("./lexicon.txt")) {
      config.model.vits.lexicon = "./lexicon.txt";
    }

    if (Exists("./dict/jieba.dict.utf8")) {
      config.model.vits.dict_dir = "./dict";
    }
  }

  if (Exists("./phone.fst") && Exists("./date.fst") && Exists("./number.fst")) {
    config.rule_fsts = "./phone.fst,./date.fst,number.fst";
  }

  if (Exists("./phone-zh.fst") && Exists("./date-zh.fst") && Exists("./number-zh.fst")) {
    config.rule_fsts = "./phone-zh.fst,./date-zh.fst,number-zh.fst";
  }

  if (Exists("./rule.far")) {
    config.rule_fars = "./rule.far";
  }

  tts_ = SherpaOnnxCreateOfflineTts(&config);
}

 CNonStreamingTextToSpeechDlg::~CNonStreamingTextToSpeechDlg() {
  if (tts_) {
    SherpaOnnxDestroyOfflineTts(tts_);
  }
  if (generate_thread_ && generate_thread_->joinable()) {
    generate_thread_->join();
  }

  if (play_thread_ && play_thread_->joinable()) {
    play_thread_->join();
  }
 }


 static std::string ToString(const CString &s) {
    CT2CA pszConvertedAnsiString(s);
    return std::string(pszConvertedAnsiString);
 }

void CNonStreamingTextToSpeechDlg::OnBnClickedOk() {
  CString s;
  speaker_id_.GetWindowText(s);
  int speaker_id = _ttoi(s);
  if (speaker_id < 0) {
    AfxMessageBox(Utf8ToUtf16("Please input a valid speaker ID").c_str(), MB_OK);
    return;
  }

  speed_.GetWindowText(s);
  float speed = static_cast<float>(_ttof(s));
  if (speed < 0) {
    AfxMessageBox(Utf8ToUtf16("Please input a valid speed").c_str(), MB_OK);
    return;
  }

  my_text_.GetWindowText(s);

  std::string ss = ToString(s);
  if (ss.empty()) {
    AfxMessageBox(Utf8ToUtf16("Please input your text").c_str(), MB_OK);
    return;
  }

  if (play_thread_) {
    g_killed = true;
    g_stopped = true;
    if (play_thread_->joinable()) {
      play_thread_->join();
    }
  }

  g_killed = false;
  g_stopped = false;
  g_started = false;
  g_buffer.samples = {};

  // Caution(fangjun): It is not efficient to re-create the thread. We use this approach
  // for simplicity
  play_thread_ = std::make_unique<std::thread>(StartPlayback, SherpaOnnxOfflineTtsSampleRate(tts_));

  if (generate_thread_ && generate_thread_->joinable()) {
    generate_thread_->join();
  }

  output_filename_.GetWindowText(s);
  std::string filename = ToString(s);

  generate_thread_ = std::make_unique<std::thread>([ss, this,filename, speaker_id, speed]() {
      std::string text = ss;

      // generate_btn_.EnableWindow(FALSE);

	  const SherpaOnnxGeneratedAudio *audio =
		  SherpaOnnxOfflineTtsGenerateWithCallback(tts_, text.c_str(), speaker_id, speed, &AudioGeneratedCallback);
      // generate_btn_.EnableWindow(TRUE);
       g_stopped = true;

	  int ok = SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate,
						filename.c_str());

	  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);

	  if (ok) {
		// AfxMessageBox(Utf8ToUtf16(std::string("Saved to ") + filename + " successfully").c_str(), MB_OK);

		// AppendLineToMultilineEditCtrl(my_hint_, std::string("Saved to ") + filename + " successfully");
	  } else {
		// AfxMessageBox(Utf8ToUtf16(std::string("Failed to save to ") + filename).c_str(), MB_OK);

		// AppendLineToMultilineEditCtrl(my_hint_, std::string("Failed to saved to ") + filename);
	  }
  });

  //CDialogEx::OnOK();
}

void CNonStreamingTextToSpeechDlg::OnBnClickedStop() { g_killed = true; }
