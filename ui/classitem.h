#ifndef CLASSITEM_H
#define CLASSITEM_H

#include <QObject>
#include <QGraphicsPixmapItem>
#include <QPixmap>
#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QWidget>
#include <QDebug>
#include <QMouseEvent>

class ClassItem: public QObject, public QGraphicsPixmapItem
{
    Q_OBJECT //TODO: add this may cause error, should run qmake again
public:
    QPixmap pix;
    QGraphicsEllipseItem *cross = NULL;
    int row_, col_, number_, class_idx_, extract_size_;
    int center_[2] = {0,0};
    int crosspos[2] = {0,0};
    bool selected = false;
    int type = 0; //1:pos; -1:neg
    int classK = 0;
    int cnt = 0;
    ClassItem(QPixmap p, int, int, int, int class_idx, QStringList center, int extract_size, int _classK, int _cnt);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void setCross();
    void setNumber();
};

#endif // CLASSITEM_H
