
// NonStreamingTextToSpeechDlg.cpp : implementation file
//

#include "pch.h"
#include "framework.h"
#include "NonStreamingTextToSpeech.h"
#include "NonStreamingTextToSpeechDlg.h"
#include "afxdialogex.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


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
}

BEGIN_MESSAGE_MAP(CNonStreamingTextToSpeechDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
        ON_BN_CLICKED(IDOK, &CNonStreamingTextToSpeechDlg::OnBnClickedOk)
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
    AppendLineToMultilineEditCtrl(my_hint_, "\r\n\r\nPlease input your text and click the button Generate");

}

void CNonStreamingTextToSpeechDlg::Init() {
    InitHint();
    speaker_id_.SetWindowText(Utf8ToUtf16("0").c_str());
    speed_.SetWindowText(Utf8ToUtf16("1.0").c_str());

	bool ok = true;
    std::string error_message = "--------------------";
  if (!Exists("./model.onnx")) {
    error_message += "Cannot find ./model.onnx\r\n";
    ok = false;
  }

  if (!Exists("./lexicon.txt")) {
    error_message += "Cannot find ./lexicon.txt\r\n";
    ok = false;
  }

  if (!Exists("./tokens.txt")) {
    error_message += "Cannot find ./tokens.txt\r\n";
    ok = false;
  }

  if (!ok) {
    generate_btn_.EnableWindow(FALSE);
    error_message +=
        "\r\nPlease refer to\r\n"
        "https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html";
    error_message += "\r\nto download models.\r\n";
    error_message += "\r\nWe given an example below\r\n\r\n";
    error_message +=
        "wget -O model.onnx "
        "https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/"
        "vits-aishell3.onnx\r\n";
    error_message +=
        "wget  "
        "https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/"
        "lexicon.txt\r\n";
    error_message +=
        "wget  "
        "https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/"
        "tokens.txt\r\n";

    AppendLineToMultilineEditCtrl(my_hint_, error_message);
    return;
  }

  // Now init tts
  SherpaOnnxOfflineTtsConfig config;
  memset(&config, 0, sizeof(config));
  config.model.debug = 0;
  config.model.num_threads = 1;
  config.model.provider = "cpu";
  config.model.vits.model = "./model.onnx";
  config.model.vits.lexicon = "./lexicon.txt";
  config.model.vits.tokens = "./tokens.txt";

  tts_ = SherpaOnnxCreateOfflineTts(&config);
}

 CNonStreamingTextToSpeechDlg::~CNonStreamingTextToSpeechDlg() {
  if (tts_) {
    SherpaOnnxDestroyOfflineTts(tts_);
  }
 }



void CNonStreamingTextToSpeechDlg::OnBnClickedOk() {
  // TODO: Add your control notification handler code here
  CString s;
  speaker_id_.GetWindowText(s);
  int speaker_id = _ttoi(s);
  if (speaker_id < 0) {
    AfxMessageBox(Utf8ToUtf16("Please input a valid speaker ID").c_str(), MB_OK);
    return;
  }

  speed_.GetWindowText(s);
  float speed = _ttof(s); 
  if (speed < 0) {
    AfxMessageBox(Utf8ToUtf16("Please input a valid speed").c_str(), MB_OK);
    return;
  }

  my_text_.GetWindowText(s);
  CT2CA pszConvertedAnsiString(s);
  std::string ss(pszConvertedAnsiString);
  if (ss.empty()) {
    AfxMessageBox(Utf8ToUtf16("Please input your text").c_str(), MB_OK);
    return;
  }

const SherpaOnnxGeneratedAudio *audio =
      SherpaOnnxOfflineTtsGenerate(tts_, ss.c_str(), speaker_id, speed);
  std::string filename = "./generated.wav";
int ok = SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate,
                    filename.c_str());

  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);

  if (ok) {
    AfxMessageBox(Utf8ToUtf16("Saved to ./generated.wav successfully").c_str(), MB_OK);
  } else {
    AfxMessageBox(Utf8ToUtf16("Failed to save to ./generated.wav").c_str(), MB_OK);
  }

  //CDialogEx::OnOK();
}
