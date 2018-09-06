import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{DataType, StringType}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

// The DoubleTokenizer converts our input coefficients into a DenseVector of Doubles.
// The input is something like: 58.003555 -0.07620182 -0.050569158 0.7153169
// The `createTransformFunc` takes the input String splits it, converts them into Doubles,
// and then packs them into a Vector.
class DoubleTokenizer (override val uid: String)
  extends UnaryTransformer[String, DenseVector, DoubleTokenizer] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("double"))

  override protected def createTransformFunc: String => DenseVector = input => {
    val words = input.toLowerCase.split("\\s").filter(_.nonEmpty)
    val doubles = words.map(_.toDouble)
    new DenseVector(doubles)
  }

  override protected def outputDataType: DataType = VectorType

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == StringType, s"Input type must be string type but got $inputType.")
  }

  override def copy(extra: ParamMap): DoubleTokenizer = defaultCopy(extra)
}