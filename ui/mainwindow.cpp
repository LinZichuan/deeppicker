#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QBrush>
#include <QRadialGradient>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    mrclistview = new QListView();
    graphview = new QGraphicsView();
    classview = new ClassQGraphicsView();

    buttongroup = new QButtonGroup;

    import = new QPushButton("add to list");
    buttonpick = new QPushButton("pick");
    ctf_estimation = new QPushButton("ctf-estimation");
    classify = new QPushButton("2d-classify");
    showclass = new QPushButton("showclass");
    retrain = new QPushButton("retrain");  // TODO:open a dialog to input retrain-model name
    extract = new QPushButton("extract");
    recenter = new QPushButton("recenter");
    recenter_show = new QPushButton("show recentered");
    subset = new QPushButton("extract subset");
    particle_show = new QPushButton("hide picked");
    class_button = new QPushButton("...");

    ctf_estimation->setEnabled(false);
    classify->setEnabled(false);
    showclass->setEnabled(false);
    retrain->setEnabled(false);
    extract->setEnabled(false);
    recenter->setEnabled(false);
    recenter_show->setEnabled(false);
    subset->setEnabled(false);
    particle_show->setEnabled(true);

    /*
    buttongroup->addButton(classify);
    buttongroup->addButton(showclass);
    buttongroup->addButton(retrain);
    buttongroup->addButton(extract);
    buttongroup->addButton(recenter);
    */
    train_groupbox = new QGroupBox("Train");

    QSettings settings("deeppicker.conf", QSettings::NativeFormat);
    QStringList keys = settings.allKeys();
    for (int i = 0; i < keys.size(); ++i) {
        if (verbose) qDebug() << keys[i] << "," << settings.value(keys[i]);
        qDebug() << keys[i] << "," << settings.value(keys[i]);
    }
    text_import_mrc = new QLineEdit(settings.value("import_mrc").toString());
    text_output_path = new QLineEdit(settings.value("output_path").toString());
    text_output_coor = new QLineEdit(settings.value("coor_output").toString());
    text_model = new QLineEdit(settings.value("model_path").toString());
    text_particle_size = new QLineEdit(settings.value("particle_size").toString());
    text_edge = new QLineEdit("1");
    text_pick_number = new QLineEdit("1");
    //text_threshold = new QLineEdit(settings.value("threshold").toString());
    text_threshold_spinbox = new QDoubleSpinBox(this);
    text_threshold_spinbox->setMinimum(0.0);
    text_threshold_spinbox->setMaximum(1.0);
    text_threshold_spinbox->setSingleStep(0.1);
    text_threshold_spinbox->setValue(0.5);

    text_symbol = new QLineEdit(settings.value("symbol").toString());
    text_class2d = new QLineEdit(settings.value("class2d_name").toString());
    //text_contrast = new QLineEdit(settings.value("contrast").toString());
    text_contrast_spinbox = new QDoubleSpinBox(this);
    text_contrast_spinbox->setMinimum(0.0);
    text_contrast_spinbox->setMaximum(10.0);
    text_contrast_spinbox->setSingleStep(1.0);
    text_contrast_spinbox->setValue(3.0);
    /*
    text_contrast_slider = new QSlider(this);
    text_contrast_slider->setOrientation(Qt::Horizontal);
    text_contrast_slider->setMinimum(0);
    text_contrast_slider->setMaximum(10);
    text_contrast_slider->setSingleStep(1);
    text_contrast_slider->setValue(3);
    */
    text_gpu = new QLineEdit(settings.value("gpu").toString());

    text_Voltage = new QLineEdit(settings.value("Voltage").toString());
    text_Cs = new QLineEdit(settings.value("Cs").toString());
    text_AmpCnst = new QLineEdit(settings.value("AmpCnst").toString());
    text_xmag = new QLineEdit(settings.value("XMAG").toString());
    text_DStep = new QLineEdit(settings.value("DStep").toString());
    text_ResMin = new QLineEdit(settings.value("ResMin").toString());
    text_ResMax = new QLineEdit(settings.value("ResMax").toString());
    text_dFMin = new QLineEdit(settings.value("dFMin").toString());
    text_dFMax = new QLineEdit(settings.value("dFMax").toString());
    text_FStep = new QLineEdit(settings.value("FStep").toString());
    text_dAst = new QLineEdit(settings.value("dAst").toString());
    text_angpix = new QLineEdit(settings.value("angpix").toString());

    text_iter = new QLineEdit(settings.value("iter").toString());
    text_classK = new QLineEdit(settings.value("classK").toString());
    if (text_classK->text().replace(" ", "") != "") classview->init(text_classK->text().toInt());
    text_psi_step = new QLineEdit(settings.value("psi_step").toString());
    text_offset_range = new QLineEdit(settings.value("offset_range").toString());
    text_offset_step = new QLineEdit(settings.value("offset_step").toString());

    connect(text_import_mrc, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_output_path, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_output_coor, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_model, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_particle_size, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_edge, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    //connect(text_threshold, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_threshold_spinbox, SIGNAL(valueChanged(double)), this, SLOT(text_threshold_spinbox_changed()));
    connect(text_symbol, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_class2d, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    //connect(text_contrast, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_contrast_spinbox, SIGNAL(valueChanged(double)), this, SLOT(text_contrast_spinbox_changed()));
    connect(text_gpu, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));

    connect(text_Voltage, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_Cs, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_AmpCnst, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_xmag, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_DStep, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_ResMin, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_ResMax, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_dFMin, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_dFMax, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_FStep, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_dAst, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_angpix, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));

    connect(text_iter, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_classK, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_psi_step, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_offset_range, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));
    connect(text_offset_step, SIGNAL(textChanged(const QString &)), this, SLOT(text_changed(const QString &)));

    QFile *temp = new QFile();
    if (!temp -> exists("deeppicker.conf")) {
        //temp -> mkdir("deeppicker.conf");
        text_output_coor->setText("pick-result");
        text_particle_size->setText("300");
        text_edge->setText("1");
        text_symbol->setText("_pick");
        text_class2d->setText("default_job");
        //text_contrast->setText("3");
        text_gpu->setText("0");

        text_Voltage->setText("300");
        text_Cs->setText("2.7");
        text_AmpCnst->setText("0.1");
        text_xmag->setText("10000");
        text_DStep->setText("1.32");
        text_ResMin->setText("30");
        text_ResMax->setText("5");
        text_dFMin->setText("5000");
        text_dFMax->setText("50000");
        text_FStep->setText("500");
        text_dAst->setText("100");
        text_angpix->setText("1.32");

        text_iter->setText("25");
        text_classK->setText("50");
        text_psi_step->setText("12");
        text_offset_range->setText("5");
        text_offset_step->setText("2");

        QString picker_path = QCoreApplication::applicationDirPath();
        QString model_path = picker_path;
        model_path = model_path.replace("ui", "trained_model/joint-8");
        if (verbose) {
            qDebug() << "picker_path = " << picker_path;
            qDebug() << "model_path = " << model_path;
        }
        text_model->setText(model_path);
    }

    textbutton_mrc = new QPushButton("...");
    textbutton_output = new QPushButton("...");
    textbutton_model = new QPushButton("...");
    /*
    text_import_mrc = new QLineEdit("/data00/Data/zw18/test/*.mrc");
    text_output_path = new QLineEdit("/data00/Programs/thuempicker/OUTPUT/");
    text_output_coor = new QLineEdit("pick-result/");
    text_model = new QLineEdit("/data00/Programs/thuempicker/trained_model/joint-4");
    text_particle_size = new QLineEdit("180");
    text_threshold = new QLineEdit("0.95");
    text_symbol = new QLineEdit("_pick");
    text_class2d = new QLineEdit("");
    */
    log = new QTextEdit("");

    QFormLayout *textlayout = new QFormLayout();
    QHBoxLayout *buttonlayout = new QHBoxLayout();
    QHBoxLayout *listloglayout = new QHBoxLayout();
    QVBoxLayout *left = new QVBoxLayout();
    QHBoxLayout *wholelayout = new QHBoxLayout();

    scene = new QGraphicsScene();
    classscene = new QGraphicsScene();
    pixmap = new QPixmap();
    //qimage = new QImage(data, w, h, w, QImage::Format_Grayscale8);
    qimage = new QImage();

    //mrclistview->resize(10,50);

    paint_file = ""; //TODO:path.toStdString();

    connect(mrclistview,SIGNAL(clicked(QModelIndex)),this,SLOT(itemClicked(QModelIndex)));
    connect(import,SIGNAL(clicked()),this,SLOT(importClicked()));
    connect(buttonpick,SIGNAL(clicked()),this,SLOT(pickClicked()));
    connect(ctf_estimation,SIGNAL(clicked()),this,SLOT(estimationClicked()));
    connect(classify,SIGNAL(clicked()),this,SLOT(classifyClicked()));
    connect(retrain,SIGNAL(clicked()),this,SLOT(retrainClicked()));
    connect(extract,SIGNAL(clicked()),this,SLOT(extractClicked()));
    connect(showclass,SIGNAL(clicked()),this,SLOT(showclassClicked()));
    connect(subset,SIGNAL(clicked()),this,SLOT(subsetClicked()));
    connect(recenter,SIGNAL(clicked()),this,SLOT(recenterClicked()));
    connect(recenter_show,SIGNAL(clicked()),this,SLOT(recenter_showClicked()));
    connect(particle_show,SIGNAL(clicked()),this,SLOT(particle_showClicked()));
    connect(class_button,SIGNAL(clicked()),this,SLOT(class_buttonClicked()));
    connect(textbutton_mrc,SIGNAL(clicked()),this,SLOT(textbutton_mrcClicked()));
    connect(textbutton_output,SIGNAL(clicked()),this,SLOT(textbutton_outputClicked()));
    connect(textbutton_model,SIGNAL(clicked()),this,SLOT(textbutton_modelClicked()));

    QHBoxLayout *row1 = new QHBoxLayout();
    QHBoxLayout *row2 = new QHBoxLayout();
    QHBoxLayout *row3 = new QHBoxLayout();
    row1->addWidget(text_import_mrc);
    row1->addWidget(textbutton_mrc);
    row2->addWidget(text_output_path);
    row2->addWidget(textbutton_output);
    row3->addWidget(text_model);
    row3->addWidget(textbutton_model);

    textlayout->addRow(tr("Import Mrc:"), row1);
    textlayout->addRow(tr("Output path:"), row2);
    textlayout->addRow(tr("Coor output:"), text_output_coor);
    //textlayout->addRow(tr("&Image path:"), text);
    textlayout->addRow(tr("Model path:"), row3);

    QHBoxLayout *row4 = new QHBoxLayout();
    QHBoxLayout *row5 = new QHBoxLayout();
    QFormLayout *r1  = new QFormLayout();
    QFormLayout *r2  = new QFormLayout();
    QFormLayout *r3  = new QFormLayout();
    QFormLayout *r4  = new QFormLayout();
    QFormLayout *r5  = new QFormLayout();
    QFormLayout *r6  = new QFormLayout();
    QFormLayout *r7  = new QFormLayout();

    r1->addRow(tr("Particle size(pix):"), text_particle_size);
    //r2->addRow(tr("Threshold:"), text_threshold);
    r2->addRow(tr("Threshold:"), text_threshold_spinbox);
    r3->addRow(tr("Symbol:"), text_symbol);
    r7->addRow(tr("Edge:"), text_edge);
    r4->addRow(tr("Class2d name:"), text_class2d);
    //r5->addRow(tr("Contrast(1~5):"), text_contrast);
    r5->addRow(tr("Contrast(0~10):"), text_contrast_spinbox);
    r6->addRow(tr("GPU device:"), text_gpu);

    row4->addLayout(r1);
    row4->addLayout(r2);
    row4->addLayout(r3);
    row4->addLayout(r7);

    row5->addLayout(r4);
    row5->addLayout(r5);
    row5->addLayout(r6);

    textlayout->addRow(row4);
    textlayout->addRow(row5);


    /*
    textlayout->addRow(tr("Particle size(pix):"), text_particle_size);
    //textlayout->addRow(tr("&Pick num:"), text_pick_number);
    textlayout->addRow(tr("Threshold:"), text_threshold);
    textlayout->addRow(tr("Symbol:"), text_symbol);
    textlayout->addRow(tr("Class2d name:"), text_class2d);
    textlayout->addRow(tr("Contrast(1~5):"), text_contrast);
    textlayout->addRow(tr("GPU device:"), text_gpu);
    */

    QHBoxLayout *ctf1 = new QHBoxLayout();
    QHBoxLayout *ctf2 = new QHBoxLayout();
    QHBoxLayout *ctf3 = new QHBoxLayout();
    QHBoxLayout *ctf4 = new QHBoxLayout();
    QHBoxLayout *class1 = new QHBoxLayout();
    QHBoxLayout *class2 = new QHBoxLayout();

    QFormLayout *t1  = new QFormLayout();
    QFormLayout *t2  = new QFormLayout();
    QFormLayout *t3  = new QFormLayout();
    QFormLayout *t4  = new QFormLayout();
    QFormLayout *t5  = new QFormLayout();
    QFormLayout *t6  = new QFormLayout();
    QFormLayout *t7  = new QFormLayout();
    QFormLayout *t8  = new QFormLayout();
    QFormLayout *t9  = new QFormLayout();
    QFormLayout *t10 = new QFormLayout();
    QFormLayout *t11 = new QFormLayout();
    QFormLayout *t12 = new QFormLayout();
    QFormLayout *t13 = new QFormLayout();
    QFormLayout *t14 = new QFormLayout();
    QFormLayout *t15 = new QFormLayout();
    QFormLayout *t16 = new QFormLayout();
    QFormLayout *t17 = new QFormLayout();

    t1->addRow(tr("Voltage(kV):"), text_Voltage);
    t2->addRow(tr("Cs(mm):"), text_Cs);
    t3->addRow(tr("AmpCnst:"), text_AmpCnst);
    //t4->addRow(tr("XMAG:"), text_xmag);
    //t5->addRow(tr("DStep(um):"), text_DStep);
    t6->addRow(tr("ResMin(A):"), text_ResMin);
    t7->addRow(tr("ResMax(A):"), text_ResMax);
    t8->addRow(tr("dFMin(A):"), text_dFMin);
    t9->addRow(tr("dFMax(A):"), text_dFMax);
    t10->addRow(tr("FStep(A):"), text_FStep);
    t11->addRow(tr("dAst(A):"), text_dAst);
    t12->addRow(tr("angpix:"), text_angpix);

    t13->addRow(tr("iteration(25):"), text_iter);
    t14->addRow(tr("class(50):"), text_classK);
    //t15->addRow(tr("psi_step(12):"), text_psi_step);
    //t16->addRow(tr("offset_range(5):"), text_offset_range);
    //t17->addRow(tr("offset_step(2):"), text_offset_step);
    t15->addRow(tr("Angular Search Range(6):"), text_psi_step);
    t16->addRow(tr("Offset Search Range(5):"), text_offset_range);
    t17->addRow(tr("Offset Search Step(1):"), text_offset_step);

    ctf1->addLayout(t1);
    ctf1->addLayout(t2);
    ctf1->addLayout(t3);

    ctf2->addLayout(t8);
    ctf2->addLayout(t9);
    ctf2->addLayout(t10);

    //ctf2->addLayout(t4);
    //ctf2->addLayout(t5);

    ctf3->addLayout(t6);
    ctf3->addLayout(t7);

    ctf4->addLayout(t11);
    ctf4->addLayout(t12);

    class1->addLayout(t13);
    class1->addLayout(t14);
    class1->addLayout(t15);

    class2->addLayout(t16);
    class2->addLayout(t17);

    textlayout->addRow(ctf1);
    textlayout->addRow(ctf2);
    textlayout->addRow(ctf3);
    textlayout->addRow(ctf4);
    textlayout->addRow(class1);
    textlayout->addRow(class2);
    /*
    ctf1->addWidget(tr("Voltage(kV):"));
    ctf1->addWidget(text_Voltage);
    ctf1->addWidget(tr("Cs(mm):"));
    ctf1->addWidget(text_Cs);
    ctf1->addWidget(tr("AmpCnst:"));
    ctf1->addWidget(text_AmpCnst);
    */

    /*
    ctf2->addWidget();
    ctf2->addWidget();
    ctf2->addWidget();
    ctf2->addWidget();
    ctf2->addWidget();
    ctf2->addWidget();

    ctf3->addWidget();
    ctf3->addWidget();
    ctf3->addWidget();
    ctf3->addWidget();
    ctf3->addWidget();
    ctf3->addWidget();

    ctf4->addWidget();
    ctf4->addWidget();
    ctf4->addWidget();
    ctf4->addWidget();
    ctf4->addWidget();
    ctf4->addWidget();


    textlayout->addRow(tr("Voltage(kV):"), text_Voltage);
    textlayout->addRow(tr("Cs(mm):"), text_Cs);
    textlayout->addRow(tr("AmpCnst:"), text_AmpCnst);
    textlayout->addRow(tr("XMAG:"), text_xmag);
    textlayout->addRow(tr("DStep(um):"), text_DStep);
    textlayout->addRow(tr("ResMin(A):"), text_ResMin);
    textlayout->addRow(tr("ResMax(A):"), text_ResMax);
    textlayout->addRow(tr("dFMin(A):"), text_dFMin);
    textlayout->addRow(tr("dFMax(A):"), text_dFMax);
    textlayout->addRow(tr("FStep(A):"), text_FStep);
    textlayout->addRow(tr("dAst(A):"), text_dAst);
    textlayout->addRow(tr("angpix:"), text_angpix);
    */

    textlayout->addRow(class_button);

    QVBoxLayout *pick_buttonlayout = new QVBoxLayout();
    pick_buttonlayout->addWidget(import);
    pick_buttonlayout->addWidget(buttonpick);
    pick_buttonlayout->addWidget(particle_show);
    import->setMinimumSize(40, 40);
    buttonpick->setMinimumSize(40, 40);
    particle_show->setMinimumSize(40, 40);

    QGridLayout *train_buttonlayout = new QGridLayout();
    train_buttonlayout->addWidget(ctf_estimation, 0, 0);
    train_buttonlayout->addWidget(classify, 0, 1);
    train_buttonlayout->addWidget(showclass, 1, 0);
    train_buttonlayout->addWidget(recenter, 1, 1);
    train_buttonlayout->addWidget(extract, 2, 0);
    train_buttonlayout->addWidget(retrain, 2, 1);
    train_buttonlayout->addWidget(recenter_show, 3, 0);
    train_buttonlayout->addWidget(subset, 3, 1);
    //train_buttonlayout->addWidget(retrain, 2, 0, 2, 2);
    //train_buttonlayout->addStretch(1);
    train_groupbox->setLayout(train_buttonlayout);

    buttonlayout->addLayout(pick_buttonlayout);
    buttonlayout->addWidget(train_groupbox);

    //listloglayout->addWidget(mrclistview);
    //listloglayout->addWidget(log);

    left->addLayout(textlayout);
    left->addLayout(buttonlayout);
    //left->addLayout(listloglayout);

    //left->addWidget(mrclistview);
    left->addWidget(log);

    //wholelayout->setSpacing(0);
    //wholelayout->setMargin(0);

    wholelayout->addLayout(left);
    wholelayout->addWidget(mrclistview);
    wholelayout->addWidget(graphview);
    wholelayout->addWidget(classview);

    //mrclistview->setSizePolicy(QSizePolicy::Expanding);
    ui->centralWidget->setLayout(wholelayout);

    job_picker_name = text_class2d->text();
    if (verbose) qDebug() << job_picker_name;


    if (QDir(text_output_path->text() + "/" + text_output_coor->text()).entryInfoList().size() > 0) {
        ctf_estimation->setEnabled(true);
        classify->setEnabled(true);
    }
    QString picker_path = QCoreApplication::applicationDirPath();
    QString deeppickerui_path = picker_path;
    QString thuempicker_path = deeppickerui_path.replace("ui", "");
    QString base = thuempicker_path;
    QString class2D_path = QString("%2/Class2D/%1").arg(job_picker_name).arg(text_output_path->text());
    if (QDir(class2D_path + "/allclass").exists()) {
        showclass->setEnabled(true);
        subset->setEnabled(true);
        extract->setEnabled(true);
        retrain->setEnabled(true); //TODO: retrain after extract!!!
        //recenter->setEnabled(true);
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::text_threshold_spinbox_changed()
{
    qDebug() << text_threshold_spinbox->value();
    adjusting_threshold = true;
    call_repaint = true;
    update();
}

void MainWindow::text_contrast_spinbox_changed()
{
    truncate = text_contrast_spinbox->value();
    qDebug() << truncate;
    call_repaint = true;
    update();
}

void MainWindow::text_changed(const QString &)
{
    QSettings settings("deeppicker.conf", QSettings::NativeFormat);
    //qDebug() << QVariant(text_import_mrc->text()).toString() <<"-=========================================";
    settings.setValue("import_mrc", text_import_mrc->text());
    settings.setValue("output_path", text_output_path->text());
    settings.setValue("coor_output", text_output_coor->text());
    settings.setValue("model_path", text_model->text());
    settings.setValue("particle_size", text_particle_size->text());
    settings.setValue("edge", text_edge->text());
    //settings.setValue("threshold", text_threshold->text());
    settings.setValue("symbol", text_symbol->text());
    settings.setValue("class2d_name", text_class2d->text());
    //settings.setValue("contrast", text_contrast->text());
    settings.setValue("gpu", "" + text_gpu->text() + "");

    settings.setValue("Voltage", text_Voltage->text());
    settings.setValue("Cs", text_Cs->text());
    settings.setValue("AmpCnst", text_AmpCnst->text());
    //settings.setValue("XMAG", text_xmag->text());
    //settings.setValue("DStep", text_DStep->text());
    settings.setValue("ResMin", text_ResMin->text());
    settings.setValue("ResMax", text_ResMax->text());
    settings.setValue("dFMin", text_dFMin->text());
    settings.setValue("dFMax", text_dFMax->text());
    settings.setValue("FStep", text_FStep->text());
    settings.setValue("dAst", text_dAst->text());
    settings.setValue("angpix", text_angpix->text());

    settings.setValue("iter", text_iter->text());
    settings.setValue("classK", text_classK->text());
    settings.setValue("psi_step", text_psi_step->text());
    settings.setValue("offset_range", text_offset_range->text());
    settings.setValue("offset_step", text_offset_step->text());

    //qDebug() << settings.value("import_mrc").toString() << "===========================================================";
    classview->init(text_classK->text().toInt());
    QString class2D_path = QString("%1/Class2D/%2").arg(text_output_path->text()).arg(text_class2d->text());
    if (!QDir(class2D_path + "/allclass").exists()) {
        showclass->setEnabled(false);
        subset->setEnabled(false);
        recenter->setEnabled(false);
        extract->setEnabled(false);
        //retrain->setEnabled(); //TODO: retrain after extract!!!
    } else {
        showclass->setEnabled(true);
        subset->setEnabled(true);
        recenter->setEnabled(true);
        extract->setEnabled(true);
    }

}
void MainWindow::textbutton_mrcClicked()
{
    //QString dtext = QFileDialog::getExistingDirectory();
    QString dtext = QFileDialog::getOpenFileName();
    if (verbose) qDebug() << dtext;
    text_import_mrc->setText(dtext);
}
void MainWindow::textbutton_outputClicked()
{
    QString dtext = QFileDialog::getExistingDirectory();
    //if (dtext.size() > 0) dtext = dtext + "/";
    if (verbose) qDebug() << dtext;
    text_output_path->setText(dtext);
}
void MainWindow::textbutton_modelClicked()
{
    QString dtext = QFileDialog::getOpenFileName();
    dtext = dtext.replace(".index", "").replace(".meta","").replace(".data-00000-of-00001","");
    if (verbose) qDebug() << dtext;
    text_model->setText(dtext);
}
void MainWindow::importClicked()
{
    QString text_import_mrc_text = text_import_mrc->text();
    for (int i = 0; i < 10; ++i) {
        text_import_mrc_text = text_import_mrc_text.replace("//", "/");
    }
    int pos = text_import_mrc_text.lastIndexOf("/");
    mrc_folder = text_import_mrc_text.left(pos+1);
    mrc_filter = text_import_mrc_text.right(text_import_mrc_text.size() - pos-1);
    if (verbose) qDebug() << mrc_folder << mrc_filter;

    QDir dir(mrc_folder);
    QStringList filters;
    filters << mrc_filter;
    qDebug() <<"=----------------------->>>>>>>>>>>>>>>>" << mrc_folder;
    dir.setNameFilters(filters);

    QFileInfoList list;
    QFileInfo fileInfo;
    QString title;
    QString path;
    list = dir.entryInfoList();
    int index = 0;
    QStringList strList;
    mrclist.clear();
    while (index < list.size()) {
        fileInfo = list.at(index++);
        title = fileInfo.fileName();
        path = fileInfo.absoluteFilePath();
        if (verbose) qDebug() << title << path;
        strList.append(path);
        QStandardItem *item_p;
        mrc_item ins({false, "", item_p});
        mrcmap[path.toStdString()] = ins;
        if (verbose) qDebug() << "<<<<<<<<<<<<<" << path;
        mrclist.push_back(path.toStdString());
    }

    // load picked results
    QString pick_result_dir = text_output_path->text() +"/"+ text_output_coor->text();
    QDir dir_pick(pick_result_dir); //"pick-result";
    if (verbose) qDebug() << "<<++++++?>>>>>>" << pick_result_dir;
    QStringList filters_pick;
    qDebug() <<"=----------------------->>>>>>>>>>>>>>>>" << pick_result_dir;
    qDebug() <<"=----------------------->>>>>>>>>>>>>>>>" << text_symbol->text();
    filters_pick << "*" + text_symbol->text() + ".star";
    dir_pick.setNameFilters(filters_pick);

    QFileInfoList listp = dir_pick.entryInfoList();
    int index_pick = 0;
    string symbol = text_symbol->text().toStdString(); //"_pick";
    qDebug() << "--------------------------------->>>>>>>>>>>>>>>>>>>>>>>" << listp.size();
    while (index_pick < listp.size()) {
        QString star_name = listp.at(index_pick).fileName();
        int symbol_index = star_name.lastIndexOf(QString::fromStdString(symbol));
        //if (verbose) qDebug() << QString::fromStdString(symbol) << ">>>" << star_name << "   " << symbol_index;
        qDebug() << QString::fromStdString(symbol) << ">>>" << star_name << "   " << symbol_index;
        if (symbol_index != -1) {
            //NOTE: mrc_folder must end with '/'!
            string file_name = mrc_folder.toStdString() + star_name.replace(text_symbol->text() + ".star", ".mrc").toStdString();
            if (verbose) qDebug() << ">>>>>>>>>>>>>" << QString::fromStdString(file_name);
            mrcmap[file_name].picked = true;
            mrcmap[file_name].pickresult_file = listp.at(index_pick).absoluteFilePath().toStdString();
            if (verbose) qDebug() << ">>>>>>>>>>>>>" << QString::fromStdString(mrcmap[file_name].pickresult_file);
        }
        index_pick++;
    }

    standardItemModel = new QStandardItemModel();
    int nCount = strList.size();
    for(int i = 0; i < nCount; i++)
    {
        QString string1 = static_cast<QString>(strList.at(i));
        //QStandardItem *item = new QStandardItem(string);
        QStringList string1list = string1.split('/');
        string1 = string1list[string1list.length()-1];
        string mrcpath = mrclist[i];
        mrcmap[mrcpath].item_p = new QStandardItem(string1);

        //QLinearGradient linearGrad(QPointF(0, 0), QPointF(200, 200));
        //linearGrad.setColorAt(0, Qt::darkGreen);
        //linearGrad.setColorAt(1, Qt::yellow);
        //QBrush brush(linearGrad);
        if (mrcmap[mrcpath].picked == true) {
            QBrush brush(Qt::green);
            mrcmap[mrcpath].item_p->setBackground(brush);
        } else {
            QBrush brush(Qt::yellow);
            mrcmap[mrcpath].item_p->setBackground(brush);
        }

        standardItemModel->appendRow(mrcmap[mrcpath].item_p);
    }
    mrclistview->setModel(standardItemModel);
    //mrclistview->setModelColumn(300);
    mrclistview->setFixedSize(200,700);
    /*
    mrclistview->resize(200, mrclistview->height());
    log->resize(400, log->height());
    */
    //mrclistview->setResizeMode(QListView::Adjust);
    //mrclistview->setUniformItemSizes(true);

    //mrclistview->setMinimumSize(280, 500);
    qDebug() << this->size().height() << " " << this->size().width();
    //mainwindow_origin_height = this->size().height();
    //mrclistview->setFixedSize(250, this->size().height());
    //mrclistview->setFixedSize(250, this->size().height()/2);
    //mrclistview->adjustSize();

    mrclistview->setSelectionMode(QAbstractItemView::ExtendedSelection);
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    mrclistview->resize(200, this->height());
    //mrclistview->setMinimumSize(mrclistview->width(), mrclistview->height());
    //mrclistview->setMinimumSize(280, this->height()-50);
    //mrclistview->resize(280, this->height()-50);
    qDebug() << this->size().height() << " " << this->size().width();
    //mrclistview->setFixedSize(280,500);
    //mrclistview->setFixedSize(250, this->size().height()-50);
    //mrclistview->adjustSize();
}

void MainWindow::itemClicked(QModelIndex index)
{
    // Visualize mrc thread
    paint_file = mrc_folder.toStdString() + index.data().toString().toStdString();
    if (verbose) qDebug() << QString::fromStdString(paint_file);
    pickresult_file = "";
    if (mrcmap[paint_file].picked == true) {
        pickresult_file = mrcmap[paint_file].pickresult_file;
        qDebug() << "clicked pickresult_file: " << QString::fromStdString(pickresult_file);
        if (verbose) qDebug() << "clicked pickresult_file: " << QString::fromStdString(pickresult_file);
    }
    //QFuture<void> f = QtConcurrent::run(this, &MainWindow::repaint);
    particle_show->setText("hide picked");
    particle_show_bool = true;
    call_repaint = true;
    mrclistview->resize(280, mrclistview->height());
    update();
}

void MainWindow::class_buttonClicked()
{
    QFileDialog* fd = new QFileDialog( this );
    //fd->addFilter( "Images (*.png *.jpg *.xpm)" );
    fd->show();
}
/*
void MainWindow::visualProcess()
{
    p = new QProcess(this);
    repaint();
    connect(p,SIGNAL(readyReadStandardOutput()),this,SLOT(outlog()));
    connect(p,SIGNAL(readyReadStandardError()),this,SLOT(outlog()));
    p->start("/bin/csh", options);
    p->waitForFinished(-1);
    qDebug() << "process finish, " << task_name << " successfully!";
}
*/

void MainWindow::retrainClicked()
{
    QString picker_path = QCoreApplication::applicationDirPath();
    QString deeppickerui_path = picker_path;
    QString thuempicker_path = deeppickerui_path.replace("ui", "");
    QString base = thuempicker_path; //"/data00/Programs/thuempicker";
    //QString base = "/data00/Programs/thuempicker";
    job_picker_name = text_class2d->text();
    QString class2D_path = text_output_path->text() + QString("/Class2D/%2").arg(job_picker_name);
    QString origin_model_path = text_model->text();

    //QDir *temp = new QDir();
    //if (!temp -> exists(model_dir)) {
    //    temp -> mkdir(model_dir);
    //}

    QString new_model_dir = text_output_path->text() + "/trained_model/";
    QString new_model_name = "retrained_model";

    QList<int> posList = classview->posList;
    QList<int> negList = classview->negList;
    QStringList poslist;
    QStringList neglist;

    QString sorted_class_file = class2D_path + "/sorted_class.txt";
    ifstream fin(sorted_class_file.toStdString());
    string sort_idx;
    getline(fin, sort_idx);
    fin.close();
    QString sort_idx_q = QString::fromStdString(sort_idx);
    QStringList sort_idx_list = sort_idx_q.split(',');

    for (int i = 0; i < posList.size(); ++i) {
        bool ok;
        int class_idx = sort_idx_list[posList[i]].toInt(&ok, 10);
        poslist.append(QString::number(class_idx, 10));
    }
    for (int i = 0; i < negList.size(); ++i) {
        bool ok;
        int class_idx = sort_idx_list[negList[i]].toInt(&ok, 10);
        neglist.append(QString::number(class_idx, 10));
    }

    QString pos = poslist.join(",");
    QString neg = neglist.join(",");
    //if (verbose) qDebug() << pos << neg;
    qDebug() << pos << neg;
    //TODO: extracted data path?

    QString cmd = QString("setenv LD_LIBRARY_PATH %1/softwares/cuda/lib64:$LD_LIBRARY_PATH; \
            cd %1; %1/softwares/python/python27/bin/python train.py --train_type 6 --pos_list '%4' --neg_list '%5' \
            --train_inputDir %6/extracted_data --train_inputFile '' --particle_number 20000 \
            --model_save_dir %2 --model_save_file %3 --model_load_file './trained_model/joint-8'")
                  .arg(base).arg(new_model_dir).arg(new_model_name).arg(pos).arg(neg).arg(class2D_path);
    qDebug() << cmd;

    if (verbose) qDebug() << cmd;
    QStringList options;
    options << "-c" << cmd;
    QFuture<void> f = QtConcurrent::run(this, &MainWindow::runCmdProcess, options, QString("retrain"));

}

void MainWindow::extractClicked()
{
    QString picker_path = QCoreApplication::applicationDirPath();
    QString deeppickerui_path = picker_path;
    QString thuempicker_path = deeppickerui_path.replace("ui", "");
    QString base = thuempicker_path; //"/data00/Programs/thuempicker";
    //QString base = "/data00/Programs/thuempicker";
    job_picker_name = text_class2d->text();
    QString class2D_path = QString("%2/Class2D/%1").arg(job_picker_name).arg(text_output_path->text());
    //QString cmd = QString("cd %1; \
    //    %1/softwares/python/python27/bin/python %1/script/seperate_class.py  %3/run_it025_data.star  %2/class2d_seperated_star; \
    //    cp  %2/class2d_seperated_star/*  %2/Micrographs; \
    //    cd %1; sh runextractclass.sh  %2/Micrographs  %2/extracted_data;").arg(base).arg(text_output_path->text()).arg(class2D_path);
    QString cmd = QString("cd %1; \
        %1/softwares/python/python27/bin/python %1/script/seperate_class.py  %3/run_it025_data.star  %3/class2d_seperated_star  %5; \
        cp  %3/class2d_seperated_star/*  %2/Micrographs; \
        cd %1;  %1/softwares/python/python27/bin/python extractData.py --inputDir %2/Micrographs --mrc_number 1000 \
        --coordinate_symbol '_class' --class_number %5 --particle_size %4 --save_dir %3/extracted_data --produce_negative True ;")
        .arg(base).arg(text_output_path->text()).arg(class2D_path).arg(text_particle_size->text()).arg(text_classK->text());
    //if (verbose) qDebug() << cmd;
    qDebug() << cmd;
    QStringList options;
    options << "-c" << cmd;
    QFuture<void> f = QtConcurrent::run(this, &MainWindow::runCmdProcess, options, QString("extract"));
}

void MainWindow::particle_showClicked()
{
    if (particle_show->text() == "show picked") {
        particle_show->setText("hide picked");
        particle_show_bool = true;
    }
    else if (particle_show->text() == "hide picked") {
        particle_show->setText("show picked");
        particle_show_bool = false;
    }
    call_repaint = true;
    update();
}

void MainWindow::recenter_showClicked()
{
    string suffix = ".star";
    string recenter_suffix = "_recentered.star";
    if (recenter_show->text() == "show recentered") {
        recenter_show->setText("show un-recentered");
        qDebug() << QString::fromStdString(pickresult_file);
        pickresult_file = pickresult_file.substr(0, pickresult_file.length()-suffix.length()) + "_recentered" + suffix;
        qDebug() << QString::fromStdString(pickresult_file);
    }
    else if (recenter_show->text() == "show un-recentered") {
        recenter_show->setText("show recentered");
        qDebug() << QString::fromStdString(pickresult_file);
        pickresult_file = pickresult_file.substr(0, pickresult_file.length()-recenter_suffix.length()) + suffix;
        qDebug() << QString::fromStdString(pickresult_file);
    }
    call_repaint = true;
    update();
}

void MainWindow::recenterClicked()
{
    //if (all_recentered == true) return;

    //Write adjusted center to class_center.conf
    job_picker_name = text_class2d->text();
    QString class2D_path = text_output_path->text() + "/Class2D/" + job_picker_name;
    QString class_center_file = class2D_path + "/class_center.conf";
    QSettings class_centers(class_center_file, QSettings::NativeFormat);
    //QList<QGraphicsItem*> items = classscene->items();
    if (verbose) qDebug() << "#items" << classitemlist.size();
    for (int i = 0; i < classitemlist.size(); ++i) {
        ClassItem* it = classitemlist.at(i);
        //qDebug () << "number=" << it->number_;
        it->center_[0] = it->center_[0] + it->crosspos[0];
        it->center_[1] = it->center_[1] + it->crosspos[1];
        qDebug() << it->center_[0] << " " << it->center_[1];
        QStringList qlist;
        qlist.append(QString::number(it->center_[0],10) );
        qlist.append(QString::number(it->center_[1],10) );
        class_centers.setValue( QString::number(it->class_idx_,10), qlist );
        it->setCross();
    }
    all_recentered = true;

    QString picker_path = QCoreApplication::applicationDirPath();
    QString deeppickerui_path = picker_path;
    QString thuempicker_path = deeppickerui_path.replace("ui", "");
    QString base = thuempicker_path; //"/data00/Programs/thuempicker";
    QString pick_result_dir = "pick-result";
    QString transitions = class2D_path + "/transitions.py";
    //QString cmd = QString("cd %1; python %2/script/get_recenter_transitions_of_each_class.py %6; \
    //    python %2/script/scan_run_it_data_get_recenter_star_nompi.py %3/run_it025_data.star %4 %5;")
    //    .arg(text_output_path->text()).arg(base).arg(class2D_path).arg(pick_result_dir).arg(transitions).arg(text_particle_size->text());
    QString cmd = QString("cd %1; %2/softwares/python/python27/bin/python %2/script/get_recenter_transitions_of_each_class.py %6 %7 %9; &&\
        %2/softwares/python/python27/bin/python %2/script/scan_run_it_data_get_recenter_star_nompi.py %3/run_it025_data.star %4 %5 %8 %9;")
        .arg(text_output_path->text()).arg(base).arg(class2D_path).arg(pick_result_dir).arg(transitions).arg(text_particle_size->text()).arg(class_center_file).arg(text_symbol->text()).arg(text_classK->text());
    qDebug() << cmd;
    QStringList options;
    options << "-c" << cmd;
    QFuture<void> f = QtConcurrent::run(this, &MainWindow::runCmdProcess, options, QString("recenter"));
    f.waitForFinished();

    /*
    recentered = !recentered;
    if (recentered) {
        qDebug() << "recentered";
        qDebug() << "before: " << QString::fromStdString(pickresult_file);
        int pos = pickresult_file.find_last_of("_pick");
        qDebug() << pos;
        pickresult_file = pickresult_file.replace(pos-4, pos, "_recentered.star");
        qDebug() << "after:  " << QString::fromStdString(pickresult_file);
    } else {
        qDebug() << "before: " << QString::fromStdString(pickresult_file);
        pickresult_file = mrcmap[paint_file].pickresult_file;
        qDebug() << "after:  " << QString::fromStdString(pickresult_file);
    }
    //repaint();
    call_repaint = true;
    update();
    */
}

int* MainWindow::count_class(string class2D_starfile) {
    ifstream fin(class2D_starfile);
    int *res = new int[text_classK->text().toInt()];
    memset(res, 0, text_classK->text().toInt()*4);
    string s;
    for (int i = 0; i < 33; ++i) {
        getline(fin, s);
        //qDebug() << QString::fromStdString(s);
    }
    string cnstr;
    while (fin >> s >> s >> cnstr >> s >> s >> s >> s\
             >> s >> s >> s >> s >> s >> s >> s 
             >> s >> s >> s >> s >> s >> s >> s 
             >> s >> s >> s >> s >> s >> s >> s >> s){
        int cn = QString::fromStdString(cnstr).toInt();
        res[cn-1] += 1;
        //qDebug() << QString::fromStdString(cnstr) << " " << QString::number(cn, 10);
        //if (cn == 1)
        //    qDebug() << "start from 1";
    }
    fin.close();
    return res;
}

void MainWindow::subsetClicked()
{
    QString picker_path = QCoreApplication::applicationDirPath();
    QString deeppickerui_path = picker_path;
    QString thuempicker_path = deeppickerui_path.replace("ui", "");
    QString base = thuempicker_path; //"/data00/Programs/thuempicker";
    job_picker_name = text_class2d->text();
    QString class2D_path = QString("%2/Class2D/%1").arg(job_picker_name).arg(text_output_path->text());

    QList<int> posList = classview->posList;
    QStringList poslist;
    QString sorted_class_file = class2D_path + "/sorted_class.txt";
    ifstream fin(sorted_class_file.toStdString());
    string sort_idx;
    getline(fin, sort_idx);
    fin.close();
    QString sort_idx_q = QString::fromStdString(sort_idx);
    QStringList sort_idx_list = sort_idx_q.split(',');
    for (int i = 0; i < posList.size(); ++i) {
        bool ok;
        int class_idx = sort_idx_list[posList[i]].toInt(&ok, 10);
        poslist.append(QString::number(class_idx, 10));
    }
    QString pos = poslist.join(",");
    QString pickresult = text_output_path->text() + "/" + text_output_coor->text() + "/subset";

    QString cmd = QString("cd %1;  %1/softwares/python/python27/bin/python %1/script/subset.py  %2/run_it025_data.star  %3  %4  %5;")
        .arg(base).arg(class2D_path).arg(pos).arg(pickresult).arg(text_symbol->text());
    qDebug() << cmd;
    QStringList options;
    options << "-c" << cmd;
    QFuture<void> f = QtConcurrent::run(this, &MainWindow::runCmdProcess, options, QString("subset"));
    //TODO

}

void MainWindow::showclassClicked()
{
    job_picker_name = text_class2d->text();
    QString class2D_path = QString("%2/Class2D/%1").arg(job_picker_name).arg(text_output_path->text());
    QString output_path = class2D_path + "/allclass";
    QString class2D_info = class2D_path + "/run_it025_data.star";
    int *cnt = count_class(class2D_info.toStdString());
    for (int i = 0; i < text_classK->text().toInt(); ++i){
        qDebug() << i << ":" << cnt[i];
    }
    /*
    read_class_image = true;
    QString cmd = QString("cd %1; python %1/script/read_class.py %2/run_it025_classes.mrcs %3; \
        python %1/script/sort_class_by_whitevariance.py %3").arg(text_output_path->text()).arg(class2D_path).arg(output_path);
    qDebug() << cmd;
    QStringList options;
    options << "-c" << cmd;
    QFuture<void> f = QtConcurrent::run(this, &MainWindow::runCmdProcess, options, QString("showclass"));
    f.waitForFinished();
    */
    classview->setScene(classscene);
    QString sorted_class_file = class2D_path + "/sorted_class.txt";
    ifstream fin(sorted_class_file.toStdString());
    string sort_idx;
    getline(fin, sort_idx);
    fin.close();
    QString sort_idx_q = QString::fromStdString(sort_idx);
    QStringList sort_idx_list = sort_idx_q.split(',');

    QSettings class_centers(class2D_path + "/class_center.conf", QSettings::NativeFormat);
    if (verbose) qDebug() << "HHHHHHHHHHHHHHH=" << class2D_path + "/class_center.conf";
    classitemlist.clear();
    int extract_size = text_particle_size->text().toInt() * 1.5;
    int classK = text_classK->text().toInt();
    for (int i = 0; i < classK; ++i) {
        bool ok;
        int class_idx = sort_idx_list[i].toInt(&ok, 10);
        if (verbose) qDebug() << i << ":" << class_idx;
        QString filename = QString("%1/%2.png").arg(output_path).arg(class_idx);
        qDebug() << filename;
        QImage *smallimage = new QImage();
        if (smallimage->load(filename)) {
            *smallimage = smallimage -> scaled(100, 100, Qt::KeepAspectRatio);
            int row = i / 5, col = i % 5;
            QStringList center = class_centers.value(QString::number(class_idx, 10)).toStringList();
            //qDebug () << QString::number(class_idx, 10) << ": " << center;
            ClassItem *item = new ClassItem(QPixmap::fromImage(*smallimage), row, col, i, class_idx, center, extract_size, text_classK->text().toInt(), cnt[class_idx-1]);
            item->setOffset(100*col, 100*row);
            classitemlist.append(item);
            /*
            QPen pen = item->pen();
            pen.setWidth(2);
            pen.setColor(QColor(0, 160, 230));
            item->setPen(pen);
            item->setBrush(QColor(247, 160, 57));
            */
            classscene->addItem(item);
        }
    }
    //repaint();
    classview->setMinimumSize(100*5 + 10, 100*5 + 10);
    recenter->setEnabled(true);
}

void MainWindow::chooseClicked()
{
    if (verbose) qDebug() << "clicked";
    QFileDialog *fileDialog = new QFileDialog(this);
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), "/home/jana", tr("Image Files (*.png *.jpg *.bmp)"));
}

void MainWindow::classifyClicked() //TODO: the logic here should be modified
//TODO: decouple readclass from classfiy
//TODO: readclass in firstly clicking showclass, according to the input job_picker_name!
//Intergrate 2D-classify and readclass
{
    QString picker_path = QCoreApplication::applicationDirPath();
    QString deeppickerui_path = picker_path;
    QString thuempicker_path = deeppickerui_path.replace("ui", "");
    QString base = thuempicker_path; //"/data00/Programs/thuempicker";
    if (verbose) qDebug() << "2d classify";
    //QString cmd = "cd /data00/Programs/thuempicker/data/; source /data00/System/sys_cshrc.csh; which python; ";

    //QDateTime current_date_time = QDateTime::currentDateTime();
    //QString current_date = current_date_time.toString("yyyy_MM_dd_hh_mm_ss");
    //string new_job_picker_name = "job_picker_" + current_date.toStdString();
    //job_picker_name = new_job_picker_name;
    job_picker_name = text_class2d->text();

    QString pick_result_dir = text_output_path->text() + "/" + text_output_coor->text();
    QString cmd = QString("cd %1; \
        source /data00/System/sys_cshrc.csh; cp %4/script/relion_2dclass.sh %1; ./relion_2dclass.sh %2 %3 %5 %6 %7 %8 %9 %10 %11 %12 %13;")
        .arg(text_output_path->text()).arg(job_picker_name).arg(text_symbol->text()).arg(base).arg(mrc_folder).arg(pick_result_dir).arg(text_particle_size->text()).arg("classify")
        .arg(text_iter->text()).arg(text_classK->text()).arg(text_psi_step->text()).arg(text_offset_range->text()).arg(text_offset_step->text());

    qDebug() << "2D-class>>>>>>>>>>>>>>>>>>>>>>>>>>>.";
    qDebug() << cmd;
    //input symbol, classify on whether _new or _pick

    /*ShowClass*/
    //TODO
    //job_picker_name = new_job_picker_name;
    QString class2D_path = QString("%1/Class2D/%2").arg(text_output_path->text()).arg(job_picker_name);
    QString output_path = class2D_path + "/allclass";
    cmd += QString("cd %1; %2/softwares/python/python27/bin/python %2/script/read_class.py %3/run_it025_classes.mrcs %4 %6; \
        %2/softwares/python/python27/bin/python %2/script/sort_class_by_whitevariance.py %4 %5 %6").arg(class2D_path).arg(base).arg(class2D_path).arg(output_path).arg(text_particle_size->text()).arg(text_classK->text());
    /*****/

    qDebug() << cmd;
    QStringList options;
    options << "-c" << cmd;
    QFuture<void> f = QtConcurrent::run(this, &MainWindow::runCmdProcess, options, QString("2D-classify and readclass"));

    //TODO:how to automatically load class image
    /*
    QStringList options1;
    options1 << "-c" << cmd;
    QFuture<void> f1 = QtConcurrent::run(this, &MainWindow::runCmdProcess, options1, QString("showclass"));
    */
}

void MainWindow::estimationClicked()
{
    QString picker_path = QCoreApplication::applicationDirPath();
    QString deeppickerui_path = picker_path;
    QString thuempicker_path = deeppickerui_path.replace("ui", "");
    QString base = thuempicker_path; //"/data00/Programs/thuempicker";
    if (verbose) qDebug() << "ctf estimation";
    job_picker_name = text_class2d->text();
    QString pick_result_dir = text_output_path->text() + "/" + text_output_coor->text();
    QString cmd = QString("cd %1; \
        source /data00/System/sys_cshrc.csh; cp %4/script/relion_ctf_estimation.sh %1; ./relion_ctf_estimation.sh %2 %3 %5 %6 %7 %8 \
        %9 %10 %11 %12 %13 %14 %15 %16 %17 %18 %19 %20;")
        .arg(text_output_path->text()).arg(job_picker_name).arg(text_symbol->text()).arg(base).arg(mrc_folder).arg(pick_result_dir).arg(text_particle_size->text()).arg("estimation")
        .arg(text_Voltage->text()).arg(text_Cs->text()).arg(text_AmpCnst->text())
        //.arg(text_xmag->text()).arg(text_DStep->text()).arg(text_ResMin->text())
        .arg("10000").arg(text_angpix->text()).arg(text_ResMin->text())
        .arg(text_ResMax->text()).arg(text_dFMin->text()).arg(text_dFMax->text())
        .arg(text_FStep->text()).arg(text_dAst->text()).arg(text_angpix->text()); //TODO: new_

    qDebug() << cmd;
    QStringList options;
    options << "-c" << cmd;
    QFuture<void> f = QtConcurrent::run(this, &MainWindow::runCmdProcess, options, QString("ctf estimation"));
}

vector<int> MainWindow::getSelectedIndex()
{
    vector<int> res;
    auto mod = mrclistview->selectionModel();
    QModelIndexList modlist = mod->selectedRows();
    if (mod->hasSelection()) {
        int cnt = modlist.count();
        for (int j = 0; j < cnt; ++j) {
            int row = modlist.at(j).row();
            if (verbose) qDebug() << row ;
            res.push_back(row);
        }
    }
    return res;
}

void MainWindow::chooseallClicked()
{
    mrclistview->selectAll();
}

void MainWindow::retrainMessage()
{
    retrain->setEnabled(true);
    disconnect(this, &MainWindow::retrainFinished, this, &MainWindow::retrainMessage);
    QString new_model_dir = text_output_path->text() + "/trained_model/";
    QString new_model_name = "retrained_model";
    QString new_model_path = new_model_dir + new_model_name;
    QString ori_model_path = text_model->text();
    /*
    QMessageBox *mes = new QMessageBox(this);
    mes->setWindowTitle("Model is updated.");
    QString picker_path = QCoreApplication::applicationDirPath();
    QString deeppickerui_path = picker_path;
    QString thuempicker_path = deeppickerui_path.replace("ui", "");
    QString base = thuempicker_path; //"/data00/Programs/thuempicker";
    mes->setText(QString("Original model:%1\n\nNew model:%2\n\nInitial model:%3").arg(ori_model_path).arg(new_model_path).arg(base + "/trained_model/joint-8"));
    mes->exec();
    */
    QMessageBox::StandardButton reply = QMessageBox::question(this, "Title", "Use retrained model?", QMessageBox::Yes|QMessageBox::No);
    if (reply == QMessageBox::Yes) {
        text_model->setText(new_model_path);
        if (verbose) qDebug() << "ORI:" << ori_model_path;
        if (verbose) qDebug() << "NEW:" << new_model_path;
    }
}

void MainWindow::classifyButtonEnable() {classify->setEnabled(true);}
void MainWindow::estimationButtonEnable() {ctf_estimation->setEnabled(true);}
void MainWindow::extractButtonEnable() {extract->setEnabled(true);}
void MainWindow::recenterButtonEnable() {recenter->setEnabled(true);}
void MainWindow::subsetButtonEnable() {subset->setEnabled(true);}

void MainWindow::runCmdProcess(QStringList options, const QString &task_name)
{
    p = new QProcess(this);
    QFuture<void> f = QtConcurrent::run(this, &MainWindow::outlog, p);
    if (processsignal_connected == false) {
        connect(this, SIGNAL(processFinished()), this, SLOT(outlog_last()));
        processsignal_connected = true;
    }
    if (task_name == QString("retrain") && retrainsignal_connected == false) {
        retrain->setEnabled(false);
        connect(this, SIGNAL(retrainFinished()), this, SLOT(retrainMessage()));
        retrainsignal_connected = true;
        //connect(p, SIGNAL(finished()), this, SLOT(retrainMessage()));
        //connect(p, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(retrainMessage(int a,QProcess::ExitStatus b)));
        //connect(p, SIGNAL(static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished)), this, SLOT(retrainMessage()));
        //this, SLOT(retrainMessage()));
    }
    if (task_name == QString("2D-classify and readclass")) {
        classify->setEnabled(false);
        connect(this, SIGNAL(classifyFinished()), this, SLOT(classifyButtonEnable()));
    }
    if (task_name == QString("ctf estimation")) {
        ctf_estimation->setEnabled(false);
        connect(this, SIGNAL(estimationFinished()), this, SLOT(estimationButtonEnable()));
    }
    if (task_name == QString("extract")) {
        extract->setEnabled(false);
        connect(this, SIGNAL(extractFinished()), this, SLOT(extractButtonEnable()));
    }
    if (task_name == QString("recenter")) {
        recenter->setEnabled(false);
        connect(this, SIGNAL(recenterFinished()), this, SLOT(recenterButtonEnable()));
    }
    if (task_name == QString("subset")) {
        subset->setEnabled(false);
        connect(this, SIGNAL(subsetFinished()), this, SLOT(subsetButtonEnable()));
    }
    //TODO:
    p->start("/bin/csh", options);
    //p->start(options[1]);

    p->waitForFinished(-1);

    // Output the left log
    /*
    QString abc = p->readAllStandardOutput();
    QString abc1 = p->readAllStandardError();
    qDebug() << abc << abc1;
    outlog_buffer += abc + abc1;
    log->setText(log->toPlainText() + outlog_buffer);
    outlog_buffer = "";
    QTextCursor cursor = log->textCursor();
    cursor.movePosition(QTextCursor::End);
    log->setTextCursor(cursor);
    */

    if (verbose) qDebug() << "process finish, " << task_name << " successfully!";
    if (task_name == QString("particle-pick")) {
        if (verbose) qDebug() << "repaint!";
        call_repaint = true;
        update();
    }
    //static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished)
    if (task_name == QString("retrain")) {
        //Check if retrain is successfully done.
        QString logstr = log->toPlainText();
        int logstr_length = logstr.length();
        QString tmp = logstr.mid(logstr_length - 100, 100);
        if (tmp.indexOf("<Successful>!") != -1) emitretrainFinished();
    }
    if (task_name == QString("2D-classify and readclass")) emitclassifyFinished();
    if (task_name == QString("ctf estimation")) emitestimationFinished();
    if (task_name == QString("extract")) emitextractFinished();
    if (task_name == QString("recenter")) emitrecenterFinished();
    if (task_name == QString("subset")) emitsubsetFinished();
    emitprocessFinished();
    //log->setText(log->toPlainText() + task_name + " successfully!");
}

void MainWindow::pickClicked()
{
    vector<int> selected = getSelectedIndex();
    if (selected.size() == 0) return;
    string pickfilepaths = mrclist[selected[0]];
    for (int i = 1; i < selected.size(); ++i) {
        pickfilepaths += "," + mrclist[selected[i]];
    }
    if (verbose) qDebug() << QString::fromStdString(pickfilepaths);
    //text->setText(QString::fromStdString(pickfilepaths));

    /*
    text_model = new QLineEdit("/data00/Programs/thuempicker/trained_model/joint-4");
    text_particle_size = new QLineEdit("180");
    text_pick_number = new QLineEdit("1");
    text_threshold = new QLineEdit("0.95");
    text_output_coor = new QLineEdit("pick-result");
    text_symbol = new QLineEdit("_pick");
     * */
    QString inputDir = text_import_mrc->text(); //mrc_folder; //"/data00/Data/zw18/test/";
    QString outputDir = text_output_path->text(); //"/data00/Data/zw18/test/";
    QString pickfilepath = QString::fromStdString(pickfilepaths).replace(" ", "");
    QString model = text_model->text();
    QString particle_size = text_particle_size->text(); //"180";
    QString edge = text_edge->text();
    QString picknumber = text_pick_number->text(); //"1";
    //QString threshold = text_threshold->text(); //"0.95";
    QString threshold = QString::number(text_threshold_spinbox->value(), 10, 2); //"0.95";
    QString coorOutput = text_output_coor->text(); //"pick-result";
    QString symbol = text_symbol->text(); //"_pick";
    QString curr_dir = QDir::currentPath();
    QString gpu_device = text_gpu->text().replace(" ", "");
    if (verbose) qDebug() << "pick: " << pickfilepath;
    if (verbose) qDebug() << inputDir << particle_size << threshold;
    if (verbose) qDebug() << curr_dir;

    //QSettings settings("deeppicker.conf", QSettings::NativeFormat);
    //qDebug() << QVariant(text_import_mrc->text()).toString() <<"-=========================================";
    //settings.setValue("output_path", text_output_path->text());
    //settings.setValue("import_mrc", text_import_mrc->text());
    //settings.setValue("coor_output", text_output_coor->text());
    //settings.setValue("model_path", text_model->text());
    //settings.setValue("particle_size", text_particle_size->text());
    //settings.setValue("threshold", text_threshold->text());
    //settings.setValue("symbol", text_symbol->text());/data00/Data/deeppicker_traindata/trpv1/OUTPUT
    //settings.setValue("class2d_name", text_class2d->text());
    //qDebug() << settings.value("import_mrc").toString() << "===========================================================";
    //settings.sync();

    //string cmd = "cd /data00/Programs/thuempicker/; source /data00/System/sys_cshrc.csh; which python; ";
    //cmd += "python picker.py --inputDir " + inputDir + " --pre_trained_model " + model + " --particle_size " + \
    //        particle_size + " --mrc_number " + picknumber + " --mrc_filename " + pickfilepath + \
    //        " --outputDir " + outputDir + " --coordinate_symbol " +  symbol + " --threshold " + threshold;

    //QString cmd = QString("cd /data00/Programs/thuempicker/; source /data00/System/sys_cshrc.csh; which python;\
    //               python picker.py --inputDir \"%1\" --pre_trained_model %2 --particle_size %3\
    //               --mrc_number %4 --mrc_filename %5 --outputDir %6 --coordinate_symbol %7 --threshold %8 --coorOutput %9 --deeppickerRunDir %10")
    //              .arg(inputDir).arg(model).arg(particle_size).arg(picknumber).arg(pickfilepath)
    //              .arg(outputDir).arg(symbol).arg(threshold).arg(coorOutput).arg(curr_dir);

    //inputDir, outputDir, coorOutput, particle_size, threshold, picknumber, model, symbol, deeppickerRunDir
    QString picker_path = QCoreApplication::applicationDirPath();
    QString deeppickerui_path = picker_path;
    deeppickerui_path = deeppickerui_path.replace("ui", "deeppicker-ui");
    QString cmd = deeppickerui_path + QString(" \"%1\" %2 %3 %4 %5 %6 %7 %8 %9 %10 %11 %12")
                  .arg(inputDir).arg(outputDir).arg(coorOutput).arg(particle_size).arg(threshold).arg(picknumber)
                  .arg(model).arg(symbol).arg(curr_dir).arg(gpu_device).arg(pickfilepath).arg(edge);
                  //"python picker.py --inputDir \"%1\" --pre_trained_model %2 --particle_size %3\
                  // --mrc_number %4 --mrc_filename %5 --outputDir %6 --coordinate_symbol %7 --threshold %8 --coorOutput %9 --deeppickerRunDir %10")
    qDebug() << cmd;
    QStringList options;
    options << "-c" << cmd;
    QFuture<void> f = QtConcurrent::run(this, &MainWindow::runCmdProcess, options, QString("particle-pick"));

    string rawname = paint_file.substr(paint_file.find_last_of('/')+1);
    string starname = rawname.substr(0, rawname.find_last_of(".mrc")-3) + symbol.toStdString() + ".star";
    //TODO:pickresult_file = inputDir + "/" + outputDir + "/" + starname;
    //pickresult_file = "/data00/Programs/thuempicker/" + outputDir.toStdString() + "/" + starname;

    for (int i = 0; i < selected.size(); ++i) {
        int idx = selected[i];
        string currmrc = mrclist[idx];
        string rawname1 = currmrc.substr(currmrc.find_last_of('/')+1);
        string starname1 = rawname1.substr(0, rawname1.find_last_of(".mrc")-3) + symbol.toStdString() + ".star";
        string pickresult_file1 = outputDir.toStdString() +"/"+ coorOutput.toStdString() +"/"+ starname1;
        if (verbose) qDebug() << QString::fromStdString(currmrc);
        if (verbose) qDebug() << "pickresult_file: >>>>>><<<<<<" << QString::fromStdString(pickresult_file1);

        //TODO

        mrcmap[currmrc].picked = true;
        mrcmap[currmrc].pickresult_file = pickresult_file1;
        QBrush newbrush(Qt::green);
        mrcmap[currmrc].item_p->setBackground(newbrush);
    }
    /*
    mrcmap[paint_file].picked = true;
    mrcmap[paint_file].pickresult_file = pickresult_file;
    QBrush newbrush(Qt::green);
    mrcmap[paint_file].item_p->setBackground(newbrush);
    */

    //qDebug() << QString::fromStdString(rawname);
    //qDebug() << QString::fromStdString(starname);
    //qDebug() << QString::fromStdString(pickresult_file);
    //qDebug() << QString::fromStdString(paint_file);
    //paint_file = inputDir + pickfilepath;
    //repaint();

    //call_repaint = true;
    //update();
}

void binning(float *gray, float *graybin, int row, int col, int scale) {
    int bin_row = row / scale;
    int bin_col = col / scale;
    for (int i = 0; i < bin_row; ++i) {
        for (int j = 0; j < bin_col; ++j) {
            float tmp = 0;
            for (int r = 0; r < scale; ++r) {
                for (int c = 0; c < scale; ++c) {
                    int index = (i*scale+r)*col + (j*scale+c);
                    tmp += gray[index];
                }
            }
            graybin[i*bin_col+j] = tmp;
        }
    }
}
struct param Gaussian_Distribution(float *whole, int size) {
    float mean = 0;
    for (int i = 0; i < size; ++i) {
        mean += whole[i];
    }
    mean = mean / size;
    float variance = 0;
    for (int i = 0; i < size; ++i) {
        variance += (whole[i] - mean) * (whole[i] - mean);
    }
    float standard = sqrt(variance / size);
    struct param p = {mean, standard};
    return p;
}
void grayto256(float* gray, int *bmp, float max_, float min_, int size) {
    float delta = max_ - min_;
    float ratio = 255.0f / delta;
    for (int i = 0; i < size; ++i) {
        if (gray[i] < min_) {
            bmp[i] = 0;
        } else if (gray[i] > max_) {
            bmp[i] = 255;
        } else {
            bmp[i] = int((gray[i] - min_) * ratio);
        }
    }
}
struct star_ar readstar(string starfile, bool adjusting) {
    int num = 0;
    star_point *p = new star_point[4000];
    ifstream fin(starfile);
    string s;
    for (int i = 0; i < 12; ++i) {
        fin >> s;
        //cout << s << endl;
    }
    string a, b, c;
    int ia, ib;
    double ic;
    string s1,s2,s3;
    if (adjusting) {
        while(fin >> a >> b >> c >> s1 >> s2 >> s3) {
            stringstream ss;
            ss << a;
            ss >> ia;
            stringstream ss1;
            ss1 << b;
            ss1 >> ib;
            stringstream ss2;
            ss2 << c;
            ss2 >> ic;
            p[num++] = {ia, ib, ic};
        }
    } else {
        while(fin >> a >> b >> s1 >> s2 >> s3) {
            stringstream ss;
            ss << a;
            ss >> ia;
            stringstream ss1;
            ss1 << b;
            ss1 >> ib;
            stringstream ss2;
            p[num++] = {ia, ib, 1.0};
        }
    }
    fin.close();
    struct star_ar sa = {p, num};
    return sa;
}

void MainWindow::outlog(QProcess *p)
{
    connect(p,SIGNAL(readyReadStandardOutput()),this,SLOT(outlog_thread()));
    connect(p,SIGNAL(readyReadStandardError()),this,SLOT(outlog_thread()));
    //connect(p,SIGNAL(readyReadStandardError()),this,SLOT(error_thread()));
}

void MainWindow::outlog_thread()
{
    QString abc = p->readAllStandardOutput();
    QString abc1 = p->readAllStandardError();
    if (verbose) qDebug() << abc << abc1;
    outlog_buffer += abc + abc1;
    if (outlog_buffer.size() > 50)
    {
        log->setText(log->toPlainText() + outlog_buffer);
        outlog_buffer = "";
        QTextCursor cursor = log->textCursor();
        cursor.movePosition(QTextCursor::End);
        log->setTextCursor(cursor);
    }
}
void MainWindow::outlog_last()
{
    QString abc = p->readAllStandardOutput();
    QString abc1 = p->readAllStandardError();
    if (verbose) qDebug() << abc << abc1;
    outlog_buffer += abc + abc1;
    if (abc1 != "") {
        //TODO
    }
    //log->setText(log->toPlainText() + outlog_buffer + "\nsuccessful!");
    log->setText(log->toPlainText() + outlog_buffer);
    outlog_buffer = "";
    QTextCursor cursor = log->textCursor();
    cursor.movePosition(QTextCursor::End);
    log->setTextCursor(cursor);
}
void MainWindow::paintEvent(QPaintEvent *e)
{
    if (!call_repaint) return;
    //if (classview->posList.size() > 0 && classview->negList.size() > 0)
    if (QDir(text_output_path->text() + text_output_coor->text()).entryInfoList().size() > 0) {
        ctf_estimation->setEnabled(true);
        classify->setEnabled(true);
    }
    QString picker_path = QCoreApplication::applicationDirPath();
    QString deeppickerui_path = picker_path;
    QString thuempicker_path = deeppickerui_path.replace("ui", "");
    QString base = thuempicker_path;
    QString class2D_path = QString("%2/Class2D/%1").arg(job_picker_name).arg(text_output_path->text());
    if (QDir(class2D_path + "/allclass").exists()) {
        showclass->setEnabled(true);
        subset->setEnabled(true);
        extract->setEnabled(true);
        retrain->setEnabled(true); //TODO: retrain after extract!!!
        //recenter->setEnabled(true);
    }
    if (paint_file == "") return;
    //qDebug() << "paint>>>";
    //qDebug() << QString::fromStdString(paint_file) << QString::fromStdString(pickresult_file);
    graphview->setScene(scene);
    string mode = "r";
    MRC m(paint_file.c_str(), mode.c_str());
    int row = m.getNy();
    int col = m.getNx();
    int size = row * col ;

    float *gray = new float[size];
    m.read2DIm_32bit(gray, 0);

    float *whole = new float[size];
    m.read2DIm_32bit(whole, 0);

    int binsize = 5;

    size/= (binsize*binsize);
    float *graybin = new float[size];
    binning(gray,graybin,row,col,binsize);
    float *wholebin = new float[size];
    binning(whole,wholebin,row,col,binsize);
    //Gaussian Distribution
    struct param p = Gaussian_Distribution(wholebin,size);
    delete[] whole;
    //int truncate = text_contrast->text().toFloat();
    if (verbose) qDebug() << "Truncate >>> " << truncate;
    float min_ = p.mean - truncate * p.standard; //NOTE:3 for split, 5 for view
    float max_ = p.mean + truncate * p.standard;

    int *bmp = new int[size];
    grayto256(graybin,bmp,max_,min_,size);
    col /= binsize;
    row /= binsize;

    graphview->setMinimumSize(col, row);
    //graphview->resize(col, row);

    qimage = new QImage(col, row, QImage::Format_RGB32);
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            int b = (int )*(bmp + i*col + j);
            qimage->setPixel(j, i, qRgb(b, b, b));
        }
    }
    if (pickresult_file != "" && particle_show_bool == true) {
        string pickresult_file_tmp = pickresult_file;
        if (adjusting_threshold) pickresult_file_tmp += ".all";
        star_ar star_array = readstar(pickresult_file_tmp, adjusting_threshold);
        qDebug() << QString::fromStdString(pickresult_file_tmp);
        //paint star point
        QPainter draw(this);
        int starnum = star_array.length;
        if (verbose) qDebug() << "par number=" << starnum;
        double thres = text_threshold_spinbox->value();
        qDebug() << thres;
        for (int i = 0; i < starnum; ++i) {
            //qDebug() << ">>>";
            qDebug() << adjusting_threshold;
            int x = star_array.p[i].x / binsize;
            int y = star_array.p[i].y / binsize;
            double c = star_array.p[i].c;
            if (c < thres) continue;
            int r = text_particle_size->text().toInt() / binsize / 2;
            qDebug() << x << y;
            for (int a = -r; a <= r; ++a) {
                for (int b = -r; b <= r; ++b) { //r = 22
                    if (int(a*a + b*b) > (r-1)*(r-2)  && int(a*a + b*b) <= r*r ) {
                        if ( 0 < x+a && x+a < col && 0 < y+b && y+b < row ) {
                            qimage->setPixel(x+a, y+b, qRgb(255, 255, 255));
                            //qimage->setPixel(x+a, y+b, qRgb(167, 244, 66));
                        }
                    }
                }
            }
            //qDebug() << "---";
            //(x-45,y-45), (x+45,y+45)
            //QRect rect(x, y, 3, 3);
            //draw.drawImage(rect, *qimage);
        }
    }
    scene->addPixmap(QPixmap::fromImage(*qimage));
    call_repaint = false;
    adjusting_threshold = false;
    recenter_show->setEnabled(true);
    //mrclistview->resize(280, this->height());
}
