#include "classqgraphicsview.h"

ClassQGraphicsView::ClassQGraphicsView() {}

void ClassQGraphicsView::init(int _classK)
{
    classK = _classK;
    posList.clear();
    negList.clear();
    qDebug() << classK;
    //QList<QGraphicsItem*> items = this->scene()->items();
    for (int i = 0; i < fmin(5, classK/2); ++i) {
        posList.append(i);
        //ClassItem *itempos = (ClassItem*)items.at(i);
        //itempos->type = 1;
        //ClassItem *itemneg = (ClassItem*)items.at(classK-1-i);
        //itemneg->type = -1;
    }
    //for (int i = fmax(classK-10, classK/2); i < classK; ++i) {
    //    negList.append(i);
    //}
    qDebug()<< "posList:" << posList;
    qDebug()<< "negList:" << negList;
}

void ClassQGraphicsView::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::RightButton) {
        QMenu *menu = new QMenu();
        QAction *posAction = new QAction("positive", this);
        QAction *negAction = new QAction("negative", this);
        QAction *cancelAction = new QAction("cancel", this);
        connect(posAction, SIGNAL(triggered()), this, SLOT(posClicked()));
        connect(negAction, SIGNAL(triggered()), this, SLOT(negClicked()));
        connect(cancelAction, SIGNAL(triggered()), this, SLOT(cancelClicked()));
        menu->addAction(posAction);
        menu->addAction(negAction);
        menu->addAction(cancelAction);
        menu->exec(QCursor::pos());
    }
}

QList<int> ClassQGraphicsView::getSelectedItemNumber(int type)
{
    QList<QGraphicsItem*> items = this->scene()->selectedItems();
    qDebug() << "Right Click number=" << items.size();
    QList<int> selectedItemNumber;
    for (int i = 0; i < items.size(); ++i) {
        ClassItem *item = (ClassItem*)items.at(i);
        item->type = type;
        selectedItemNumber.push_back(item->number_);
        qDebug () << "item_number:" << item->number_;
    }
    return selectedItemNumber;
}

void ClassQGraphicsView::posClicked()
{
    QList<QGraphicsItem*> items = this->scene()->items();
    qDebug() << "pos number=" << posList.size();
    posList.append(getSelectedItemNumber(1));
    posList = posList.toSet().toList();
    for (int i = 0; i < posList.size(); ++i) {
        negList.removeOne(posList[i]);
        ClassItem *item = (ClassItem*)items.at(classK-1 - posList[i]);
        //qDebug() << posList[i] << ":" << item->class_idx_;
        qDebug() << item->class_idx_;
    }
    qDebug()<< "posList:" << posList;
    qDebug()<< "negList:" << negList;
}

void ClassQGraphicsView::negClicked()
{
    QList<QGraphicsItem*> items = this->scene()->items();
    qDebug() << "neg number=" << negList.size();
    negList.append(getSelectedItemNumber(-1));
    negList = negList.toSet().toList();
    for (int i = 0; i < negList.size(); ++i) {
        posList.removeOne(negList[i]);
        ClassItem *item = (ClassItem*)items.at(classK-1 - negList[i]);
        //qDebug() << negList[i] << ":" << item->class_idx_;
        qDebug() << item->class_idx_;
    }
    qDebug()<< "posList:" << posList;
    qDebug()<< "negList:" << negList;
}

void ClassQGraphicsView::cancelClicked()
{
    qDebug() << "cancel clicked";
    QList<int> select = getSelectedItemNumber(0);
    for (int i = 0; i < select.size(); ++i) {
        negList.removeOne(select[i]);
        posList.removeOne(select[i]);
    }
    qDebug()<< "posList:" << posList;
    qDebug()<< "negList:" << negList;
}
