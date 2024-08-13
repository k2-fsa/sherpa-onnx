unit Unit1;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls;

type

  { TForm1 }

  TForm1 = class(TForm)
    StartBtn: TButton;
    SelectFileDlg: TOpenDialog;
    SelectFileBtn: TButton;
    FileNameEdt: TEdit;
    procedure FileNameEdtChange(Sender: TObject);
    procedure SelectFileBtnClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private

  public

  end;

var
  Form1: TForm1;

implementation

{$R *.lfm}

{ TForm1 }

procedure TForm1.FormCreate(Sender: TObject);
begin
  StartBtn.Enabled := False;
  SelectFileDlg.Filter := 'All Files|*.wav';
end;

procedure TForm1.SelectFileBtnClick(Sender: TObject);
begin
  if SelectFileDlg.Execute then
    begin
      FileNameEdt.Text := SelectFileDlg.FileName;
    end;
end;

procedure TForm1.FileNameEdtChange(Sender: TObject);
begin
  if FileExists(FileNameEdt.Text) then
    StartBtn.Enabled := True
  else
    StartBtn.Enabled := False;
end;

end.

