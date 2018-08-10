#ifndef CLASSQGRAPHICSSCENE_H
#define CLASSQGRAPHICSSCENE_H
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <classitem.h>
#include <QDebug>
#include <QGraphicsSceneMouseEvent>

class ClassQGraphicsScene: public QGraphicsScene
{
public:
    ClassQGraphicsScene();
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
};

#endif // CLASSQGRAPHICSSCENE_H
