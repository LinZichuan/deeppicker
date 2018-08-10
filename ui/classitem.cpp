#include "classitem.h"

ClassItem::ClassItem(QPixmap p, int row, int col, int number, int class_idx, QStringList center, int extract_size, int _classK, int _cnt)
    :QGraphicsPixmapItem(p), QObject()
{
    classK = _classK;
    cnt = _cnt;
    pix = p;
    row_ = row, col_ = col;
    number_ = number;
    class_idx_ = class_idx;
    extract_size_ = extract_size;
    bool ok;
    center_[0] = center[0].toInt(&ok, 10);
    center_[1] = center[1].toInt(&ok, 10);
    //qDebug () << "center_:" << center_[0] << "," << center_[1];

    if (number_ < fmin(5, classK/2)) type = 1;
    //else if (number_ >= fmax(classK-10, classK/2)) type = -1;

    //setFlag(QGraphicsItem::ItemIsMovable, true);
    //setFlag(QGraphicsItem::ItemIsFocusable, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setCross();
    setNumber();
}

void ClassItem::setCross()
{
    if (cross != NULL) delete cross;
    int r = 10;
    cross = new QGraphicsEllipseItem(col_*100 + center_[0]*100/extract_size_ - r, row_*100 + center_[1]*100/extract_size_ - r, r*2, r*2, this);
    cross->setFlags(QGraphicsEllipseItem::ItemIsSelectable | QGraphicsEllipseItem::ItemIsMovable);
    QPen pp = cross->pen();
    pp.setWidth(2);
    pp.setColor(QColor(160, 200, 0));
    cross->setPen(pp);
}

void ClassItem::setNumber()
{
    QGraphicsTextItem *textitem = new QGraphicsTextItem(QString::number(cnt, 10), this);
    textitem->setPos(col_*100, row_*100);
    textitem->setDefaultTextColor(QColor(160, 200, 0));

    QGraphicsTextItem *id = new QGraphicsTextItem("classid:"+QString::number(class_idx_, 10), this);
    id->setPos((col_+1)*100-80, (row_+1)*100-20);
    id->setDefaultTextColor(QColor(160, 200, 0));
}

void ClassItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    if (cross != NULL) {
        QPointF pos = cross->pos();
        crosspos[0] = int( float(pos.x())/100.0*extract_size_ );
        crosspos[1] = int( float(pos.y())/100.0*extract_size_ );
    }
    painter->drawPixmap(col_*100, row_*100, pix);
    int bor = 3;
    if (option->state & QStyle::State_Selected) {
        selected = true;
        qDebug() << "selected = " << selected;
        QColor qc = QColor(0,0,255);
        QPen pen(qc);
        pen.setWidth(3);
        painter->setPen(pen);
        //QBrush brush(QColor(255, 0, 0));
        //painter->setBrush(brush);
        painter->drawRect(col_*100+bor, row_*100+bor, 100-2*bor, 100-2*bor);
    }
    else {
        if (type == 1) {
            QPen pen(QColor(0, 255, 0));
            pen.setWidth(3);
            painter->setPen(pen);
            painter->drawRect(col_*100+bor, row_*100+bor, 100-2*bor, 100-2*bor);
        }
        else if (type == -1) {
            QPen pen(QColor(255, 0, 0));
            pen.setWidth(3);
            painter->setPen(pen);
            painter->drawRect(col_*100+bor, row_*100+bor, 100-2*bor, 100-2*bor);
        }
    }
    /*
    if (selected) {
    }
    */
}

void ClassItem::mousePressEvent(QGraphicsSceneMouseEvent *event) //TODO: why here do not respond!
{
    /*
    selected = !selected;
    qDebug() << "selected = " << selected;
    this->update();
    */
}
