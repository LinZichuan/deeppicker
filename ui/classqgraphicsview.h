#ifndef CLASSQGRAPHICSVIEW_H
#define CLASSQGRAPHICSVIEW_H
#include <QGraphicsView>
#include <QGraphicsItem>
#include <classitem.h>
#include <QDebug>
#include <QMouseEvent>
#include <QGraphicsSceneMouseEvent>
#include <fstream>
#include <QMenu>

class ClassQGraphicsView: public QGraphicsView
{
    Q_OBJECT

public:
    QList<int> posList;
    QList<int> negList;
    int classK = 0;

    ClassQGraphicsView();
    void init(int);
    void mousePressEvent(QMouseEvent *event);
    QList<int> getSelectedItemNumber(int type);
    //void mouseDoubleClickEvent(QMouseEvent *event);

public slots:
    void posClicked();
    void negClicked();
    void cancelClicked();

};

#endif // CLASSQGRAPHICSVIEW_H
