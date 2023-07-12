
// StreamingSpeechRecognitionDlg.cpp : implementation file
//

#include "pch.h"
#include "framework.h"
#include "StreamingSpeechRecognition.h"
#include "StreamingSpeechRecognitionDlg.h"
#include "afxdialogex.h"

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

// CStreamingSpeechRecognitionDlg dialog



CStreamingSpeechRecognitionDlg::CStreamingSpeechRecognitionDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_STREAMINGSPEECHRECOGNITION_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CStreamingSpeechRecognitionDlg::DoDataExchange(CDataExchange* pDX)
{
  CDialogEx::DoDataExchange(pDX);
  DDX_Control(pDX, IDOK, my_btn_);
  DDX_Control(pDX, IDC_EDIT1, my_text_);
}

BEGIN_MESSAGE_MAP(CStreamingSpeechRecognitionDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CStreamingSpeechRecognitionDlg::OnBnClickedOk)
END_MESSAGE_MAP()


// CStreamingSpeechRecognitionDlg message handlers

BOOL CStreamingSpeechRecognitionDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	SetWindowText(_T("Real-time speech recogntion with Next-gen Kaldi"));

	return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CStreamingSpeechRecognitionDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
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

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CStreamingSpeechRecognitionDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CStreamingSpeechRecognitionDlg::OnBnClickedOk() {
  // TODO: Add your control notification handler code here
  CDialogEx::OnOK();
}
