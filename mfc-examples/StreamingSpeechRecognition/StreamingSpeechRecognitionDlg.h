
// StreamingSpeechRecognitionDlg.h : header file
//

#pragma once
#include "portaudio.h"

class Microphone {
 public:
  Microphone();
  ~Microphone();
};

// CStreamingSpeechRecognitionDlg dialog
class CStreamingSpeechRecognitionDlg : public CDialogEx
{
// Construction
public:
	CStreamingSpeechRecognitionDlg(CWnd* pParent = nullptr);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_STREAMINGSPEECHRECOGNITION_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
    afx_msg void OnBnClickedOk();
    CButton my_btn_;
    CEdit my_text_;
};
