#include "classqgraphicsscene.h"

ClassQGraphicsScene::ClassQGraphicsScene()
{

}

void ClassQGraphicsScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    /*
    if (event->button() == Qt::RightButton) {
        QList<QGraphicsItem*> items = this->selectedItems();
        qDebug() << "Right Click..." << items.size();
        QList<int> selectedItemNumber;
        for (int i = 0; i < items.size(); ++i) {
            ClassItem *item = (ClassItem*)items.at(i);
            selectedItemNumber.push_back(item->number_);
            qDebug () << i;
        }
    }
    */
}
