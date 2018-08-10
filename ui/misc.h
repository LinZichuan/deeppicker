#ifndef MISC_H
#define MISC_H

#include <QMainWindow>
#include <QListView>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QModelIndex>

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QPixmap>
#include <QImage>
#include <QString>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QPushButton>
#include <QButtonGroup>
#include <QGroupBox>
#include <QTextEdit>
#include <QStringList>
#include <QFileInfoList>
#include <QFileInfo>
#include <QDir>
#include <QProcess>
#include <QPaintDevice>
#include <QDateTime>
#include <QtConcurrent>
#include <QMessageBox>
#include <QGraphicsPixmapItem>
#include <classqgraphicsview.h>
//#include <classqgraphicsscene.h>
#include <classitem.h>
#include <QFileDialog>
#include <QDoubleSpinBox>
//#include <QSizePolicy>

#include "mrc.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include <cstring>

using namespace std;

struct param{
    float mean;
    float standard;
};
struct star_point {
    int x;
    int y;
    double c;
};
struct star_ar {
    star_point* p;
    int length;
};
class Misc
{
public:
    Misc();
};



#endif // MISC_H
