#ifndef MAINWINDOW_H
#define MAINWINDOW_H


#include "misc.h"
#include <QLabel>
#include <QLineEdit>
#include <QFormLayout>
#include <vector>
#include <map>

using namespace std;

namespace Ui {
class MainWindow;
}
struct mrc_item {
    bool picked;
    string pickresult_file;
    QStandardItem *item_p;
};


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void createQimage(string, string);
    void paintEvent(QPaintEvent *e);
    void resizeEvent(QResizeEvent *e);
    int* count_class(string);

private:
    Ui::MainWindow *ui;
    bool call_repaint = false;
    QListView *mrclistview;
    map<string, mrc_item> mrcmap;
    vector<string> mrclist;
    QGraphicsView *graphview;
    ClassQGraphicsView *classview;
    QGraphicsScene *scene;
    QGraphicsScene *classscene;
    QPixmap *pixmap;
    QImage * qimage;
    QStandardItemModel *standardItemModel;

    QPushButton *import;
    QButtonGroup *buttongroup;
    QGroupBox *train_groupbox;
    QPushButton *buttonpick;
    QPushButton *chooseall;
    QPushButton *ctf_estimation;
    QPushButton *classify;
    QPushButton *showclass;
    QPushButton *extract_subset;
    QPushButton *retrain;
    QPushButton *extract;
    QPushButton *recenter;
    QPushButton *subset;
    QPushButton *recenter_show;
    QPushButton *particle_show;
    bool particle_show_bool = true;
    bool all_recentered = false;
    bool recentered = false;
    bool read_class_image = false;
    bool class_recentered = false;
    bool adjusting_threshold = false;
    QLineEdit *text;
    QTextEdit *log;
    QProcess *p;
    string paint_file = "";
    string pickresult_file = "";
    QString job_picker_name;// = "job_picker_2017_10_29_12_03_38"; //"job_picker_2017_10_25_11_09_05"; //TODO
    QString outlog_buffer = "";
    bool retrainsignal_connected = false;
    bool processsignal_connected = false;
    QList<ClassItem*> classitemlist;
    bool verbose = true;
    int mainwindow_origin_height = 0;
    double truncate = 3.0;

    QLineEdit *text_import_mrc;
    QString mrc_folder = "";
    QString mrc_filter = "";
    QLineEdit *text_output_path;
    QLineEdit *text_output_coor;
    QLineEdit *text_model;
    QLineEdit *text_particle_size;
    QLineEdit *text_pick_number;
    QLineEdit *text_edge;
    //QLineEdit *text_threshold;
    QDoubleSpinBox *text_threshold_spinbox;
    QLineEdit *text_symbol;
    QLineEdit *text_class2d;
    //QLineEdit *text_contrast;
    QDoubleSpinBox *text_contrast_spinbox;
    QLineEdit *text_gpu;

    QLineEdit *text_Voltage;
    QLineEdit *text_Cs;
    QLineEdit *text_AmpCnst;
    QLineEdit *text_xmag;
    QLineEdit *text_DStep;
    QLineEdit *text_ResMin;
    QLineEdit *text_ResMax;
    QLineEdit *text_dFMin;
    QLineEdit *text_dFMax;
    QLineEdit *text_FStep;
    QLineEdit *text_dAst;
    QLineEdit *text_angpix;

    QLineEdit *text_iter;
    QLineEdit *text_classK;
    QLineEdit *text_psi_step;
    QLineEdit *text_offset_range;
    QLineEdit *text_offset_step;

    QPushButton *class_button;
    QPushButton *textbutton_mrc;
    QPushButton *textbutton_output;
    QPushButton *textbutton_model;

public slots:
    void itemClicked(QModelIndex index);
    void chooseClicked();
    void pickClicked();
    void importClicked();
    void chooseallClicked();
    void estimationClicked();
    void classifyClicked();
    void retrainClicked();
    void extractClicked();
    void showclassClicked();
    void subsetClicked();
    void recenterClicked();
    void recenter_showClicked();
    void particle_showClicked();
    void class_buttonClicked();
    void textbutton_mrcClicked();
    void textbutton_outputClicked();
    void textbutton_modelClicked();
    void text_changed(const QString&);
    void text_threshold_spinbox_changed();
    void text_contrast_spinbox_changed();

    vector<int> getSelectedIndex();
    void runCmdProcess(QStringList options, const QString &task_name);
    void outlog(QProcess* p);
    void outlog_thread();
    void outlog_last();
    void retrainMessage();
    void classifyButtonEnable();
    void estimationButtonEnable();
    void extractButtonEnable();
    void recenterButtonEnable();
    void subsetButtonEnable();

    void emitretrainFinished() {retrainFinished();}
    void emitprocessFinished() {processFinished();}
    void emitclassifyFinished() {classifyFinished();}
    void emitestimationFinished() {estimationFinished();}
    void emitextractFinished() {extractFinished();}
    void emitrecenterFinished() {recenterFinished();}
    void emitsubsetFinished() {subsetFinished();}
signals:
    void retrainFinished();
    void processFinished();
    void classifyFinished();
    void estimationFinished();
    void extractFinished();
    void recenterFinished();
    void subsetFinished();
};

#endif // MAINWINDOW_H
