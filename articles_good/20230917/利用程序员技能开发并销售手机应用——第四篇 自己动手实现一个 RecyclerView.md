
作者：禅与计算机程序设计艺术                    

# 1.简介
  

RecyclerView 是 Android 官方推出的可滚动视图组件，其最大的特点就是灵活性高、性能好。今天，我们将通过自己的例子，一步步地深入分析 RecyclerView 的内部机制，进而自己实现一个简单的 RecyclerView。当然，这只是本系列的第一篇文章。

一般来说，RecyclerView 有如下几种用途:

1.列表显示： RecyclerView 可以用来展示多行条目信息，如网格布局中的图片列表；
2.局部刷新： RecyclerView 提供了局部刷新机制，可以快速更新 RecyclerView 中的某些条目，使得界面看起来更加流畅；
3.懒加载： RecyclerView 提供了懒加载机制，即滑到底部才开始加载下一页数据，可以有效提升界面响应速度；
4.自定义ViewHolder： RecyclerView 提供了自定义 ViewHolder 来定制化 ViewHolder 类型，从而实现复杂的布局效果；
5.HeaderView 和 FooterView： RecyclerView 支持 HeaderView 和 FooterView ，能够方便地为 RecyclerView 添加顶部和底部的固定元素。

为了更好的理解 RecyclerView 如何工作，本文将分以下五个小节进行介绍:

1.recyclerview 的内部结构及绘制流程
2.recyclerview 中的 adapter 的作用及自定义 adapter 的过程
3.recyclerview 的缓存机制
4.自定义 recyclerview item layout
5.自己实现一个简单 recycleview 例子。

# 2.recyclerview 的内部结构及绘制流程

recyclerview 整体结构如上图所示，主要包括三大块，分别是：
1. RecycledViewPool : 为RecyclerViews提供复用的 ViewHolder 。
2. LayoutManager : 根据 RecyclerView 的LayoutManager 属性，对 ViewHolder 进行排版。
3. Adapter : 用于绑定 ViewHolder 数据并负责 RecyclerView 的各种事件回调。

## 2.1 ViewCache 使用方式
当一个 RecyclerView 发生滑动或者重绘时，如果某个 ViewHolder 没有被回收掉，那么该ViewHolder的控件状态就会保留在ViewCache中。这样的话当RecyclerView再次滑动回来的时候，就不需要重新创建这个ViewHolder，而是直接从缓存中获取，从而达到了ViewHolder的复用。同时，由于ViewHolder存储的是已经渲染完成的View，因此也减少了重新渲染的开销。

## 2.2 ItemDecoration
ItemDecoration 是 RecyclerView 提供的一个装饰器类，它的作用是给 RecyclerView 的子项（item）添加额外的装饰，比如为它们添加 padding、margin、divider、background等，并且ItemDecoration可以被LayoutManager所共用。

对于 LinearLayoutManager，它的默认ItemDecorations只有一条DividerDecoration，它主要用于给 RecyclerView 中每一项之间增加 divider。

## 2.3 findViewHolderForPosition 方法
findViewHolderForPosition 方法主要是根据position参数值获取 ViewHolder 对象。在 RecyclerView.Adapter 的子类中，RecyclerView 会先调用此方法查找 ViewHolder ，如果没有找到，则会重新构建一个新的 ViewHolder ，并且把它存储在 RecyclerView 的 mScrapHeap （scrap heap 是 RecyclerView 维护的一个空闲 view holder 集合）。因此，findViewHolderForPosition 方法的运行时间会受到 mScrapHeap 大小的影响。但是，RecyclerView 在判断是否需要重新构建 ViewHolder 时，会先查看该 ViewHolder 是否已经存在于 ViewCache 中。如果存在，则从 ViewCache 中获取该 ViewHolder ，否则重新构建一个新的 ViewHolder 。

# 3.adapter 的作用及自定义 adapter 的过程

## 3.1 Recyclerview.Adapter<VH>
Recyclerview.Adapter 是 RecyclerView 的基础类，继承自 RecyclerView.Adapter 抽象类。其主要作用是控制 RecyclerView 的数据集，并提供 ViewHolder 管理和配置相关的方法。其定义如下：

```java
public abstract class Adapter<VH extends ViewHolder> {
    /**
     * Called when RecyclerView needs a new {@link ViewHolder} of the given type to represent
     * an item.
     * <p>
     * This new ViewHolder should be constructed with a new View that can represent the items
     * of the specified type. You can either create a new View manually or inflate it from an XML
     * file.
     * <p>
     * The RecyclerView will bind the ViewHolder to the corresponding item in the data set.
     */
    @NonNull public abstract VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType);

    /**
     * Called by RecyclerView to display the data at the specified position. This method should
     * update the contents of the {@link ViewHolder#itemView} to reflect the item at the given
     * position.
     * <p>
     * Note that unlike {@link ListView}, RecyclerView will not call this method
     * again if the position of the item changes unless the item itself is invalidated or the
     * new position cannot be determined. For this reason, you should only use the position
     * parameter while acquiring the related data item inside this method and do not keep a
     * copy of it anywhere else. If you need the position of an item later on (e.g. in a click
     * listener), use {@link ViewHolder#getLayoutPosition()} which will have the updated
     * position.
     */
    public void onBindViewHolder(@NonNull VH holder, int position){}
    
    //...
    
    /**
     * Returns the total number of items in the data set held by the adapter.
     *
     * @return The total number of items in this adapter.
     */
    public abstract int getItemCount();
}
```

其中，ViewHolder 是 RecyclerView 对数据的一种封装，用于 ViewHolder 的绑定与显示。

```java
public static abstract class ViewHolder {
    final View itemView;

    public ViewHolder(View itemView) {
        if (itemView == null) {
            throw new IllegalArgumentException("itemView may not be null");
        }
        this.itemView = itemView;
    }

    /**
     * Return the underlying {@link View} for this {@link ViewHolder}.
     */
    @NonNull public final View getItemView() {
        return itemView;
    }
}
```

通过这种方式，RecyclerView 就可以动态地绑定适配器的数据，并在屏幕上展示出来。具体的工作流程如下：

1. RecyclerView 请求适配器提供 ViewHolder 。
2. 如果之前没有过 ViewHolder ，适配器则会通过 findViewById 或 inflater 创建 View ，并设置点击监听器等相关属性。
3. 设置完毕后，ViewHolder 将与当前条目的对应关系绑定。
4. 当 RecyclerView 需要显示某个位置的数据时，通过 getItem 方法获得数据，并告诉 ViewHolder 更新显示。
5. ViewHolder 通过 findViewById 获取对应的控件，然后根据数据设置控件的显示内容，最后通过 post 执行动画。

由此可知， RecyclerView.Adapter 是 RecyclerView 最重要的抽象类之一。其作用是定义了一个统一的接口，通过该接口，外部可以通过 RecyclerView.Adapter 对 RecyclerView 数据进行控制，包括 ViewHolder 的创建、绑定、配置等。通过自定义 RecyclerView.Adapter ，就可以实现各种类型的 RecyclerView 效果。

## 3.2 CustomAdapter
自定义 RecyclerView.Adapter 的步骤如下：

1. 定义一个继承自 RecyclerView.Adapter 的类。
2. 实现 onCreateViewHolder 方法，该方法返回一个新的 ViewHolder。
3. 实现 onBindViewHolder 方法，该方法接收一个 ViewHolder 对象和一个条目的索引，用来设置 ViewHolder 的显示内容。
4. 返回 getItemCount 方法，该方法返回 RecyclerView 中数据集的总个数。

例如，创建一个只包含 TextView 的 Adapter ，可以参考如下代码：

```java
public class TextAdapter extends RecyclerView.Adapter<TextAdapter.TextViewHolder>{

    private List<String> dataSet;

    public TextAdapter(List<String> dataSet){
        this.dataSet = dataSet;
    }

    @NonNull
    @Override
    public TextViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.text_list_item, parent, false);
        return new TextViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull TextViewHolder holder, int position) {
        String text = dataSet.get(position);
        holder.textView.setText(text);
    }

    @Override
    public int getItemCount() {
        return dataSet.size();
    }


    public static class TextViewHolder extends RecyclerView.ViewHolder{

        TextView textView;

        public TextViewHolder(@NonNull View itemView) {
            super(itemView);

            textView = itemView.findViewById(R.id.tv_item_text);
        }
    }

}
```

这里 TextAdapter 继承了 RecyclerView.Adapter 并重写了 onCreateViewHolder 和 onBindViewHolder 方法。onCreateViewHolder 方法接受父 ViewGroup 参数和类型，返回一个新的 TextViewHolder 对象；onBindViewHolder 方法接受 TextViewholder 对象和当前条目的索引作为参数，并设置 textView 对象的文本内容。getItemCount 方法返回数据集的总个数。TextViewHolder 是对 RecyclerView 中的每个条目的 ViewHolder ，包含一个 textView 对象。在使用时，只需声明一个 RecyclerView 对象，设置 adapter 属性即可。

# 4.recyclerview 的缓存机制
RecyclerView 提供两种缓存机制，一种是 RecyclerView 本身的缓存机制，另一种是 ViewHolder 的缓存机制。
## 4.1 recycledviewpool
 RecyclerView 在 RecyclerViewPool 中维护了一组 ViewHolder ，这些 ViewHolder 可以被重复利用。也就是说， RecyclerView 会尽量复用 ViewHolder 以避免频繁创建新对象。Recyclerview 会在适配器的数据变化时清空缓存池，确保 ViewHolder 不出现内存泄漏。RecyclerView 默认会维护一个 ViewPool ，即一个 RecycledViewPool 对象。Recyclerview 还提供了设置 ViewPool 大小的方法。

```java
mRecycledViewPool = new RecycledViewPool();
recyclerView.setRecycledViewPool(mRecycledViewPool);
```

## 4.2 viewholdercache
第二种缓存机制是在 ViewHolderCache 中。 RecyclerView 不会在每次绑定数据时都去重新创建一个 ViewHolder ，而是会首先检查是否之前有 ViewHolder 处于缓存池中。如果有，则直接绑定 ViewHolder ，否则就重新创建。RecyclerView 默认情况下使用的 ViewHolderCache 是 LruCache ，但也可以通过设置参数改变缓存的大小。

```java
recyclerView.setItemViewCacheSize(DEFAULT_CACHE_SIZE);
```

# 5.自定义 recyclerview item layout

RecyclerView 提供了自定义 layout 的功能，可以在运行时动态修改 RecyclerView 的 item layout。自定义 item layout 的过程如下：

1. 在 XML 文件中定义 item layout，并为其指定 ID。
2. 在适配器中复写 getItemViewType 方法，该方法返回不同类型的 layout ID。
3. 在 onCreateViewHolder 方法中，根据不同的 viewType 指定不同的 item layout 。
4. 在 onBindViewHolder 方法中，设置 item layout 上的控件。

```xml
<?xml version="1.0" encoding="utf-8"?>
<RelativeLayout xmlns:android="http://schemas.android.com/apk/res/android"
                android:layout_width="match_parent"
                android:layout_height="wrap_content">

    <ImageView
        android:id="@+id/iv_image"
        android:layout_width="100dp"
        android:layout_height="100dp"/>

    <TextView
        android:id="@+id/tv_title"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:paddingLeft="10dp"
        android:paddingRight="10dp"
        android:gravity="center_vertical" />

</RelativeLayout>
```

```java
public class MyAdapter extends RecyclerView.Adapter<MyAdapter.CustomHolder>{

    private List<String> images;
    private List<String> titles;

    public MyAdapter(List<String> images, List<String> titles) {
        this.images = images;
        this.titles = titles;
    }

    @NonNull
    @Override
    public CustomHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view;
        switch (viewType){
            case TYPE_IMAGE:
                view = LayoutInflater.from(parent.getContext()).inflate(R.layout.image_layout, parent, false);
                break;
            case TYPE_TITLE:
                view = LayoutInflater.from(parent.getContext()).inflate(R.layout.title_layout, parent, false);
                break;
            default:
                view = LayoutInflater.from(parent.getContext()).inflate(R.layout.default_layout, parent, false);
                break;
        }
        return new CustomHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull CustomHolder holder, int position) {
        int type = getType(position);
        switch (type){
            case TYPE_IMAGE:
                Glide.with(holder.imageView.getContext())
                       .load(images.get(position))
                       .into(holder.imageView);
                break;
            case TYPE_TITLE:
                holder.titleView.setText(titles.get(position));
                break;
            default:
                break;
        }
    }

    @Override
    public int getItemCount() {
        return Math.max(images.size(), titles.size());
    }

    @Override
    public int getItemViewType(int position) {
        if(position >= getImageCount()){
            return TYPE_TITLE;
        }else{
            return TYPE_IMAGE;
        }
    }

    public int getImageCount(){
        return images!= null? images.size() : 0;
    }

    public static class CustomHolder extends RecyclerView.ViewHolder{

        ImageView imageView;
        TextView titleView;

        public CustomHolder(@NonNull View itemView) {
            super(itemView);

            imageView = itemView.findViewById(R.id.iv_image);
            titleView = itemView.findViewById(R.id.tv_title);
        }
    }

    private int getType(int position){
        if(position >= getImageCount()){
            return TYPE_TITLE;
        }else{
            return TYPE_IMAGE;
        }
    }

    private static final int TYPE_IMAGE = 0;
    private static final int TYPE_TITLE = 1;
    private static final int DEFAULT_TYPE = -1;


}
```

在该示例中，MyAdapter 重写了 getItemViewType 方法，在 onCreateViewHolder 方法中根据不同的 viewType 分别加载不同的 layout ，在 onBindViewHolder 方法中设置 layout 上面的控件。这里我自定义了两个 layout，分别是 image_layout 和 title_layout ，将 RecyclerView 中的每个条目都设置为不同的 viewType ，从而切换不同的 layout 。