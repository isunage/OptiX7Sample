#include <TestXMLConfig.h>
//For XML
#include <tinyxml2.h>
//For Parcer
#include <boost/spirit/include/qi.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/algorithm/string/replace.hpp>
//For Matrix
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <optional>
#include <string>
#include <memory>
#include <utility>
#include <cassert>
#include <vector>
#include <array>
#include <filesystem>
#include <variant>
namespace mitsuba_loader
{
	namespace qi = boost::spirit::qi;
	using namespace std::string_literals;
	struct GrammarSceneVersion : qi::grammar < std::string::iterator, std::vector<int>(), qi::locals< std::vector<int>>, qi::ascii::space_type> 
	{
		GrammarSceneVersion() : base_type(rule_start) {
			rule_start = qi::repeat(2)[qi::int_ >> qi::lit(".")] >> qi::int_;
		}
		qi::rule< std::string::iterator, std::vector<int>(), qi::locals< std::vector<int>>, qi::ascii::space_type> rule_start;
	};
	struct GrammarFloat3Value       : qi::grammar < std::string::iterator, std::vector<float>(), qi::locals< std::vector<float>>, qi::ascii::space_type>
	{
		GrammarFloat3Value() : base_type(rule_start) {
			rule_start = qi::repeat(2)[qi::float_ >> -(qi::lit(","))] >> qi::float_;
		}
		qi::rule< std::string::iterator, std::vector<float>(), qi::locals< std::vector<float>>, qi::ascii::space_type> rule_start;
	};
	struct GrammarMatrixValue       : qi::grammar < std::string::iterator, std::vector<float>(), qi::locals< std::vector<float>>, qi::ascii::space_type>
	{
		GrammarMatrixValue() : base_type(rule_start) {
			rule_start = qi::repeat(16)[qi::float_];
		}
		qi::rule< std::string::iterator, std::vector<float>(), qi::locals< std::vector<float>>, qi::ascii::space_type> rule_start;
	};
	struct GrammarSpectrumValue : qi::grammar< std::string::iterator, std::variant<std::vector<std::vector<float>>, std::vector<float>, float>(), qi::locals<std::variant<std::vector<std::vector<float>>, std::vector<float>, float>>, qi::ascii::space_type >
	{
	private:
		using FloatPairList = std::vector<std::vector<float>>;
		using FloatList = std::vector<float>;
		using Float = float;
	public:
		GrammarSpectrumValue() :base_type(rule_start) {
			rule_float = qi::float_;
			rule_float_array = (+(rule_float >> qi::lit(",")) >> rule_float);
			rule_float_pair = (rule_float >> qi::lit(":") >> rule_float);
			rule_float_pair_array = (*(rule_float_pair >> qi::lit(",")) >> rule_float_pair);
			rule_start = rule_float_array | rule_float_pair_array | rule_float;
		}
		qi::rule<std::string::iterator, std::variant<FloatPairList, FloatList, Float>(), qi::locals<std::variant<FloatPairList, FloatList, Float>>, qi::ascii::space_type > rule_start;
		qi::rule<std::string::iterator, float(), qi::locals<float>, qi::ascii::space_type>                                                                                  rule_float;
		qi::rule<std::string::iterator, std::vector<float>(), qi::locals<std::vector<float>>, qi::ascii::space_type>                                                        rule_float_pair;
		qi::rule<std::string::iterator, std::vector<std::vector<float>>(), qi::locals< std::vector<std::vector<float>>>, qi::ascii::space_type>                             rule_float_pair_array;
		qi::rule<std::string::iterator, std::vector<float>(), qi::locals< std::vector<float>>, qi::ascii::space_type>                                                       rule_float_array;
	};
	enum class SupportType
	{
		Unknown = 0,
		Float,
		Integer,
		Boolean,
		String,
		RGBColor,
		SRGBColor,
		Spectrum,
		Vector,
		Point,
		Transform,
		Animation,
		Reference,
		
		Shape,
		Bsdf,
		Texture,
		Subsurface,
		Medium,
		Phase,
		Volume,
		Emitter,
		Sensor,
		Integrator,
		Sampler,
		Film,
		Rfilter,
		Count,
	};
	auto   ToSupportObjectType(const std::string& propertyType) ->SupportType {
		SupportType objectType = SupportType::Unknown;
		if (propertyType == "bsdf") {
			objectType = mitsuba_loader::SupportType::Bsdf;
		}
		else if (propertyType == "texture") {
			objectType = mitsuba_loader::SupportType::Texture;
		}
		else if (propertyType == "subsurface") {
			objectType = mitsuba_loader::SupportType::Subsurface;
		}
		else if (propertyType == "medium") {
			objectType = mitsuba_loader::SupportType::Medium;
		}
		else if (propertyType == "phase") {
			objectType = mitsuba_loader::SupportType::Phase;
		}
		else if (propertyType == "volume") {
			objectType = mitsuba_loader::SupportType::Volume;
		}
		else if (propertyType == "emitter") {
			objectType = mitsuba_loader::SupportType::Emitter;
		}
		else if (propertyType == "sensor") {
			objectType = mitsuba_loader::SupportType::Sensor;
		}
		else if (propertyType == "integrator") {
			objectType = mitsuba_loader::SupportType::Integrator;
		}
		else if (propertyType == "sampler") {
			objectType = mitsuba_loader::SupportType::Sampler;
		}
		else if (propertyType == "film") {
			objectType = mitsuba_loader::SupportType::Film;
		}
		else if (propertyType == "rfilter") {
			objectType = mitsuba_loader::SupportType::Rfilter;
		}
		else {
			objectType = mitsuba_loader::SupportType::Unknown;
		}
		return objectType;
	}
	using  Float   = float;
	using  Integer = int64_t;
	using  Boolean = bool;
	using  String  = std::string;
	using  Float3  = std::array<float, 3>;
	enum class IntentType
	{
		None,
		Reflectance,
		Illuminant
	};
	struct RGBColor {
	public:
		RGBColor() :m_value{}, m_intent{IntentType::None}{}
		RGBColor(const RGBColor&) = default;
		RGBColor& operator=(const RGBColor&) = default;
		auto GetFloat3()const -> const Float3& {
			return m_value;
		}
		void SetFloat3(const Float3& colorValue) {
			m_value = colorValue;
		}
		auto GetIntent()const ->IntentType { return m_intent; }
		void SetIntent(IntentType intent) { m_intent = intent; }
		~RGBColor() {}
	private:
		Float3     m_value;
		IntentType m_intent;
	};
	struct SRGBColor 
	{
	public:
		SRGBColor() :m_value{ Float3{} }, m_intent{ IntentType::None }{}
		SRGBColor(const SRGBColor&) = default;
		SRGBColor& operator=(const SRGBColor&) = default;
		bool IsValue ()const { return std::get_if<0>(&m_value) != nullptr; }
		bool IsString()const { return std::get_if<1>(&m_value) != nullptr; }
		auto GetFloat3()const -> std::optional<Float3> {
			if (IsValue()) { return std::get<0>(m_value); }
			return std::nullopt;
		}
		void SetFloat3(const Float3& value) {
			m_value = value;
		}
		auto GetString()const -> std::optional<String> {
			if (IsString()) { return std::get<1>(m_value); }
			return std::nullopt;
		}
		void SetString(const String& string) {
			m_value = string;
		}
		auto GetIntent()const ->IntentType { return m_intent; }
		void SetIntent(IntentType intent) { m_intent = intent; }
		~SRGBColor() {}
	private:
		std::variant< Float3, String> m_value;
		IntentType m_intent;
	};
	struct Spectrum
	{
	public:
		Spectrum() :m_WaveLengthes{ }, m_WaveWeights{ 0.0f }, m_intent{ IntentType::None }{}
		Spectrum(const Spectrum&) = default;
		Spectrum& operator=(const Spectrum&) = default;
		bool IsUniformed()const noexcept {
			return m_WaveLengthes.size() == 0 && m_WaveWeights.size() == 1;
		}
		//Get
		auto GetSize()        const -> size_t { return m_WaveLengthes.size(); }
		auto GetWaveLengthes()const -> const std::vector<float>& { return m_WaveLengthes; }
		auto GetWaveWeights ()const -> const std::vector<float>& { return m_WaveWeights;  }
		auto GetIntent()const ->IntentType { return m_intent; }
		//Set
		void SetValue(Float uniformValue)
		{
			m_WaveLengthes.clear();
			m_WaveWeights = std::vector<float>{ {uniformValue} };
		}
		void SetValue(const std::vector<float>& lengthes, const std::vector<float>& weights)
		{
			m_WaveLengthes = lengthes;
			m_WaveWeights  = weights;
		}
		void SetValue(const std::vector<float>& weights , Float minWaveLength = 360, Float maxWaveLength = 830)
		{
			auto dL = (maxWaveLength - minWaveLength) / static_cast<float>(weights.size());
			for (auto i = 0; i < weights.size(); ++i)
			{
				m_WaveLengthes[i] = minWaveLength + dL * i;
 			}
			m_WaveWeights = weights;
		}
		void SetIntent(IntentType intent) { m_intent = intent; }
		~Spectrum() {}
	private:
		std::vector<float> m_WaveLengthes;
		std::vector<float> m_WaveWeights;
		IntentType         m_intent;
	};
	struct Vector
	{
		float x;
		float y;
		float z;
	};
	using  Point = Vector;
	struct Transform
	{
		glm::mat4x4 matrix = glm::identity<glm::mat4x4>();
	};
	struct Animation
	{
		std::vector<std::pair<float,Transform>> transforms;
	};
	struct Reference {
		String objectID;
	};
	class  Object;
	using  ObjectPtr = std::shared_ptr<Object>;
	class  Properties
	{
	private:
	public:
		Properties() {}
		bool Has(const String& name)const { return m_PropertyTypes.count(name) > 0; }
		auto GetType(const String& name)const -> SupportType {
			return m_PropertyTypes.at(name);
		}
		//Float
		void AddFloat(const String& name) {
			m_PropertyTypes[name] = SupportType::Float;
			m_Floats[name] = 0.0f;
		}
		auto GetFloat(const String& name)const-> Float {
			return m_Floats.at(name);
		}
		void SetFloat(const String& name, Float value) {
			m_PropertyTypes[name] = SupportType::Float;
			m_Floats[name] = value;
		}
		//Integer
		void AddInteger(const String& name) {
			m_PropertyTypes[name] = SupportType::Integer;
			m_Integers[name] = 0;
		}
		auto GetInteger(const String& name)const-> Integer {
			return m_Integers.at(name);
		}
		void SetInteger(const String& name, Integer value) {
			m_PropertyTypes[name] = SupportType::Integer;
			m_Integers[name] = value;
		}
		//Boolean
		void AddBoolean(const String& name) {
			m_PropertyTypes[name] = SupportType::Boolean;
			m_Booleans[name] = 0;
		}
		auto GetBoolean(const String& name)const-> Boolean {
			return m_Booleans.at(name);
		}
		void SetBoolean(const String& name, Boolean value) {
			m_PropertyTypes[name] = SupportType::Boolean;
			m_Booleans[name] = value;
		}
		//String
		void AddString(const String& name) {
			m_PropertyTypes[name] = SupportType::String;
			m_Strings[name] = {};
		}
		auto GetString(const String& name)const-> String {
			return m_Strings.at(name);
		}
		void SetString(const String& name, String value) {
			m_PropertyTypes[name] = SupportType::String;
			m_Strings[name] = value;
		}
		//RGBColor
		void AddRGBColor(const String& name)
		{
			m_PropertyTypes[name] = SupportType::RGBColor;
			m_RGBColors.at(name) = {};
		}
		auto GetRGBColor(const String& name)const-> const RGBColor& {
			return m_RGBColors.at(name);
		}
		void SetRGBColor(const String& name, const RGBColor& value) {
			m_PropertyTypes[name] = SupportType::RGBColor;
			m_RGBColors[name] = value;
		}
		//SRGBColor
		void AddSRGBColor(const String& name)
		{
			m_PropertyTypes[name] = SupportType::SRGBColor;
			m_SRGBColors.at(name) = {};
		}
		auto GetSRGBColor(const String& name)const-> const SRGBColor& {
			return m_SRGBColors.at(name);
		}
		void SetSRGBColor(const String& name, const SRGBColor& value) {
			m_PropertyTypes[name] = SupportType::SRGBColor;
			m_SRGBColors[name] = value;
		}
		//Spectrum
		void AddSpectrum(const String& name)
		{
			m_PropertyTypes[name] = SupportType::Spectrum;
			m_Spectrums.at(name) = {};
		}
		auto GetSpectrum(const String& name)const-> const Spectrum& {
			return m_Spectrums.at(name);
		}
		void SetSpectrum(const String& name, const Spectrum& value) {
			m_PropertyTypes[name] = SupportType::Spectrum;
			m_Spectrums[name] = value;
		}
		//Vector
		void AddVector(const String& name)
		{
			m_PropertyTypes[name] = SupportType::Vector;
			m_Vectors.at(name) = {};
		}
		auto GetVector(const String& name)const-> const Vector& {
			return m_Vectors.at(name);
		}
		void SetVector(const String& name, const Vector& value) {
			m_PropertyTypes[name] = SupportType::Vector;
			m_Vectors[name] = value;
		}
		//Point
		void AddPoint(const String& name)
		{
			m_PropertyTypes[name] = SupportType::Point;
			m_Points.at(name) = {};
		}
		auto GetPoint(const String& name)const-> const Point& {
			return m_Points.at(name);
		}
		void SetPoint(const String& name, const Point& value) {
			m_PropertyTypes[name] = SupportType::Point;
			m_Points[name] = value;
		}
		//Transform
		void AddTransform(const String& name)
		{
			m_PropertyTypes[name] = SupportType::Transform;
			m_Transforms.at(name) = Transform();
		}
		auto GetTransform(const String& name)const ->const Transform& {
			return m_Transforms.at(name);
		}
		void SetTransform(const String& name, const Transform& value)
		{
			m_PropertyTypes[name] = SupportType::Transform;
			m_Transforms[name] = value;
		}
		//Animation
		void AddAnimation(const String& name)
		{
			m_PropertyTypes[name] = SupportType::Animation;
			m_Animations.at(name) = Animation();
		}
		auto GetAnimation(const String& name)const ->const Animation& {
			return m_Animations.at(name);
		}
		void SetAnimation(const String& name, const Animation& value)
		{
			m_PropertyTypes[name] = SupportType::Animation;
			m_Animations[name] = value;
		}
		//Reference
		void AddReference(const String& name)
		{
			m_PropertyTypes[name] = SupportType::Reference;
			m_References.at(name) = Reference();
		}
		auto GetReference(const String& name)const ->const Reference& {
			return m_References.at(name);
		}
		void SetReference(const String& name, const Reference& value)
		{
			m_PropertyTypes[name] = SupportType::Reference;
			m_References[name]    = value;
		}
		//ObjectPtr
		auto GetObjectPtr(const String& name)const ->const ObjectPtr&;
		void SetObjectPtr(const String& name, const ObjectPtr& value);
		~Properties() {}
	private:
		std::unordered_map<String, SupportType>  m_PropertyTypes = {};
		std::unordered_map<String, Float>        m_Floats = {};
		std::unordered_map<String, Integer>      m_Integers = {};
		std::unordered_map<String, Boolean>      m_Booleans = {};
		std::unordered_map<String, String>       m_Strings = {};
		std::unordered_map<String, RGBColor>     m_RGBColors = {};
		std::unordered_map<String, SRGBColor>    m_SRGBColors = {};
		std::unordered_map<String, Spectrum>     m_Spectrums = {};
		std::unordered_map<String, Vector>       m_Vectors = {};
		std::unordered_map<String, Point>        m_Points = {};
		std::unordered_map<String, Transform>    m_Transforms = {};
		std::unordered_map<String, Animation>    m_Animations = {};
		std::unordered_map<String, Reference>    m_References = {};
		std::unordered_map<String, ObjectPtr>    m_ObjectPtrs = {};
	};
	class  Object {
	private:
		using ptr_type = std::shared_ptr<Object>;
	public:
		Object(const SupportType& objectType, const String& pluginName, const String& objectID) {
			m_ObjectType = objectType;
			m_PluginName = pluginName;
			m_ObjectID   = objectID;
		}
		auto GetObjectType()const noexcept -> SupportType {
			return m_ObjectType;
		}
		auto GetPluginName()const noexcept -> String {
			return m_PluginName;
		}
		auto GetObjectID()const noexcept ->String {
			return m_ObjectID;
		}
		void SetObjectID(const String& objectID) {
			m_ObjectID = objectID;
		}
		auto GetProperties()const noexcept -> const Properties&;
		auto GetProperties()noexcept ->             Properties&;
		void AddNestObject(const ptr_type& objectPtr);
		auto GetNestObject(size_t idx)const noexcept -> ptr_type;
		auto GetNestObjects()const -> const std::vector<ptr_type>&;
		auto GetNestObjects()->std::vector<ptr_type>&;
		virtual ~Object() {}
	private:
		SupportType           m_ObjectType  = SupportType::Unknown;
		String                m_PluginName  = "";
		String                m_ObjectID    = "";
		Properties            m_Properties  = {};
		std::vector<ptr_type> m_NestObjects = {};
	};
	void   Object::AddNestObject(const ptr_type& objectPtr) {
		m_NestObjects.push_back(objectPtr);
	}
	auto   Object::GetNestObject(size_t idx)const noexcept -> ptr_type {
		return m_NestObjects[idx];
	}
	auto   Object::GetNestObjects()const -> const std::vector<Object::ptr_type>& { return m_NestObjects; }
	auto   Object::GetNestObjects()      ->       std::vector<Object::ptr_type>& { return m_NestObjects; }
	auto   Object::GetProperties()const noexcept -> const Properties& { return m_Properties; }
	auto   Object::GetProperties()noexcept ->             Properties& { return m_Properties; }
	auto   Properties::GetObjectPtr(const String& name)const ->const ObjectPtr& {
		return m_ObjectPtrs.at(name);
	}
	void   Properties::SetObjectPtr(const String& name, const ObjectPtr& value) {
		m_PropertyTypes[name] = value->GetObjectType();
		m_ObjectPtrs[name]    = value;
	}
	class  PropertyLoader
	{
	public:
		PropertyLoader() :m_RootPath(), m_GrammarFloat3(), m_GrammarMatrix(), m_GrammarSpectrum() {}
		PropertyLoader(const String& rootPath) :PropertyLoader() {
			m_RootPath = rootPath;
		}
		~PropertyLoader() {}
		void SetRootPath(const String& rootPath) {
			m_RootPath = rootPath;
		}
		auto GetRootPath()const ->String {
			return m_RootPath;
		}
		bool LoadFloat(const tinyxml2::XMLElement* element, std::string& name, Float& value)const
		{
			auto attrib_name = element->FindAttribute("name");
			auto attrib_value = element->FindAttribute("value");
			if (attrib_name && attrib_value)
			{
				name = attrib_name->Value();
				value = attrib_value->FloatValue();
				return true;
			}
			return false;
		}
		bool LoadInteger(const tinyxml2::XMLElement* element, std::string& name, Integer& value)const
		{
			auto attrib_name = element->FindAttribute("name");
			auto attrib_value = element->FindAttribute("value");
			if (attrib_name && attrib_value)
			{
				name = attrib_name->Value();
				value = attrib_value->Int64Value();
				return true;
			}
			return false;
		}
		bool LoadBoolean(const tinyxml2::XMLElement* element, std::string& name, Boolean& value)const
		{
			auto attrib_name = element->FindAttribute("name");
			auto attrib_value = element->FindAttribute("value");
			if (attrib_name && attrib_value)
			{

				name = attrib_name->Value();
				value = attrib_value->BoolValue();
				return true;
			}
			return false;
		}
		bool LoadString(const tinyxml2::XMLElement* element, std::string& name, String& value)const
		{
			//(std::string(element->Name()) == "string");
			auto attrib_name = element->FindAttribute("name");
			auto attrib_value = element->FindAttribute("value");
			if (attrib_name && attrib_value)
			{
				name = attrib_name->Value();
				value = attrib_value->Value();
				return true;
			}
			return false;
		}
		bool LoadRGB(const tinyxml2::XMLElement* element, std::string& name, RGBColor& value)const
		{
			auto attrib_name = element->FindAttribute("name");
			auto attrib_value = element->FindAttribute("value");
			if (attrib_name && attrib_value)
			{
				name = attrib_name->Value();
				std::string input = attrib_value->Value();
				auto first = input.begin();
				auto last = input.end();
				std::vector<float> arr;
				bool result = qi::phrase_parse(first, last, m_GrammarFloat3, qi::ascii::space, arr);
				if (first == last && result) {
					value.SetFloat3(Float3{ arr[0] ,arr[1],arr[2] });
					value.SetIntent(IntentType::None);
					{
						auto attrib_intent = element->FindAttribute("intent");
						if (attrib_intent) {
							std::string intent = attrib_intent->Value();
							if (intent == "reflectance")
							{
								value.SetIntent(IntentType::Reflectance);
							}
							if (intent == "Illuminant")
							{
								value.SetIntent(IntentType::Illuminant);
							}
						}
					}
					return true;
				}
			}
			return false;
		}
		bool LoadSRGB(const tinyxml2::XMLElement* element, std::string& name, SRGBColor& value)const
		{
			//(std::string(element->Name()) == "string");
			auto attrib_name = element->FindAttribute("name");
			auto attrib_value = element->FindAttribute("value");
			if (attrib_name && attrib_value)
			{
				name = attrib_name->Value();
				std::string input = attrib_value->Value();
				auto first = input.begin();
				auto last = input.end();
				std::vector<float> arr;
				bool result = qi::phrase_parse(first, last, m_GrammarFloat3, qi::ascii::space, arr);
				if (first == last && result) {
					value.SetFloat3(Float3{ arr[0] ,arr[1],arr[2] });
					value.SetIntent(IntentType::None);
					{
						auto attrib_intent = element->FindAttribute("intent");
						if (attrib_intent) {
							std::string intent = attrib_intent->Value();
							if (intent == "reflectance")
							{
								value.SetIntent(IntentType::Reflectance);
							}
							if (intent == "Illuminant")
							{
								value.SetIntent(IntentType::Illuminant);
							}
						}
					}
					return true;
				}
				else
				{
					value.SetString(input);
					value.SetIntent(IntentType::None);
					{
						auto attrib_intent = element->FindAttribute("intent");
						if (attrib_intent) {
							std::string intent = attrib_intent->Value();
							if (intent == "reflectance")
							{
								value.SetIntent(IntentType::Reflectance);
							}
							if (intent == "Illuminant")
							{
								value.SetIntent(IntentType::Illuminant);
							}
						}
					}
					return true;
				}
			}
			return false;
		}
		bool LoadSpectrum(const tinyxml2::XMLElement* element, std::string& name, Spectrum& value)const
		{
			auto attrib_name = element->FindAttribute("name");
			auto attrib_value = element->FindAttribute("value");
			if (attrib_name && attrib_value)
			{
				name = attrib_name->Value();
				std::string input = attrib_value->Value();
				auto first = input.begin();
				auto last = input.end();
				std::variant<std::vector<std::vector<float>>, std::vector<float>, float> val;
				bool result = qi::phrase_parse(first, last, m_GrammarSpectrum, qi::ascii::space, val);
				if (first == last && result) {
					auto p_val0 = std::get_if<0>(&val);
					auto p_val1 = std::get_if<1>(&val);
					auto p_val2 = std::get_if<2>(&val);
					if (p_val0)
					{
						std::vector<float> lengthes;
						std::vector<float> weights;
						lengthes.reserve(p_val0->size());
						weights.reserve(p_val0->size());
						for (auto i = 0; i < p_val0->size(); ++i)
						{
							lengthes[i] = (*p_val0)[i][0];
							weights[i] = (*p_val0)[i][1];
						}
						value.SetValue(lengthes, weights);
					}
					if (p_val1)
					{
						value.SetValue(*p_val1);
					}
					if (p_val2)
					{
						value.SetValue(*p_val2);
					}
					value.SetIntent(IntentType::None);
					{
						auto attrib_intent = element->FindAttribute("intent");
						if (attrib_intent) {
							std::string intent = attrib_intent->Value();
							if (intent == "reflectance")
							{
								value.SetIntent(IntentType::Reflectance);
							}
							if (intent == "Illuminant")
							{
								value.SetIntent(IntentType::Illuminant);
							}
						}
					}
				}
				else {
					std::vector<float> lengthes;
					std::vector<float> weights;
					std::ifstream file(m_RootPath + input);
					std::string sentence;
					while (std::getline(file, sentence))
					{
						float l, w;
						std::stringstream ss;
						ss << sentence;
						ss >> l >> w;
						lengthes.push_back(l);
						weights.push_back(w);
					}
					value.SetValue(lengthes, weights);
					value.SetIntent(IntentType::None);
					{
						auto attrib_intent = element->FindAttribute("intent");
						if (attrib_intent) {
							std::string intent = attrib_intent->Value();
							if (intent == "reflectance")
							{
								value.SetIntent(IntentType::Reflectance);
							}
							if (intent == "Illuminant")
							{
								value.SetIntent(IntentType::Illuminant);
							}
						}
					}
				}
				return true;
			}
			return false;
		}
		bool LoadVector(const tinyxml2::XMLElement* element, std::string& name, Vector& value)const
		{
			auto attrib_name = element->FindAttribute("name");
			auto attrib_x = element->FindAttribute("x");
			auto attrib_y = element->FindAttribute("y");
			auto attrib_z = element->FindAttribute("z");
			if (attrib_name)
			{
				name = attrib_name->Value();
				value.x = attrib_x ? attrib_x->FloatValue() : 0.0f;
				value.y = attrib_y ? attrib_y->FloatValue() : 0.0f;
				value.z = attrib_z ? attrib_z->FloatValue() : 0.0f;
				return true;
			}
			return false;
		}
		bool LoadTransform(const tinyxml2::XMLElement* element, std::string& name, Transform& value)const
		{
			auto attrib_name = element->FindAttribute("name");
			if (attrib_name)
			{
				name = attrib_name->Value();
				ImplLoadTransform(element, value);
				return true;
			}
			return false;
		}
		bool LoadAnimation(const tinyxml2::XMLElement* element, std::string& name, Animation& value)const
		{
			auto attrib_name = element->FindAttribute("name");
			if (attrib_name)
			{
				name = attrib_name->Value();
				value.transforms.clear();
				auto child = element->FirstChildElement();
				while (child) {
					auto attrib_time = child->FindAttribute("time");
					auto time = attrib_time->FloatValue();
					Transform t;
					ImplLoadTransform(child, t);
					value.transforms.push_back({ time,t });
					child = child->NextSiblingElement();
				}
				return false;
			}
			return true;
		}
		bool LoadReference(const tinyxml2::XMLElement* element, std::string& name, Reference& value)const {
			//(std::string(element->Name()) == "string");
			auto attrib_name = element->FindAttribute("name");
			auto attrib_id = element->FindAttribute("id");
			if (attrib_name && attrib_id)
			{
				name = attrib_name->Value();
				value.objectID = attrib_id->Value();
				return true;
			}
			return false;
		}
		bool LoadProperties(const tinyxml2::XMLElement* element, Properties& properties)const {
			std::string propertyType = element->Value();
			if (propertyType == "float") {
				String name;
				Float  value;
				if (LoadFloat(element, name, value)) {
					properties.SetFloat(name, value);
					return true;
				}
			}
			if (propertyType == "integer") {
				String  name;
				Integer value;
				if (LoadInteger(element, name, value)) {
					properties.SetInteger(name, value);
					return true;
				}
			}
			if (propertyType == "boolean") {
				String  name;
				Boolean value;
				if (LoadBoolean(element, name, value)) {
					properties.SetBoolean(name, value);
					return true;
				}
			}
			if (propertyType == "string") {
				String name;
				String value;
				if (LoadString(element, name, value)) {
					properties.SetString(name, value);
					return true;
				}
			}
			if (propertyType == "rgb") {
				String name;
				RGBColor value;
				if (LoadRGB(element, name, value)) {
					properties.SetRGBColor(name, value);
					return true;
				}
			}
			if (propertyType == "srgb") {
				String name;
				SRGBColor value;
				if (LoadSRGB(element, name, value)) {
					properties.SetSRGBColor(name, value);
					return true;
				}
			}
			if (propertyType == "spectrum") {
				String name;
				Spectrum value;
				if (LoadSpectrum(element, name, value)) {
					properties.SetSpectrum(name, value);
					return true;
				}
			}
			if (propertyType == "vector") {
				String name;
				Vector value;
				if (LoadVector(element, name, value)) {
					properties.SetVector(name, value);
					return true;
				}
			}
			if (propertyType == "point") {
				String name;
				Point value;
				if (LoadVector(element, name, value)) {
					properties.SetPoint(name, value);
					return true;
				}
			}
			if (propertyType == "transform") {
				String name;
				Transform value;
				if (LoadTransform(element, name, value)) {
					properties.SetTransform(name, value);
					return true;
				}
			}
			if (propertyType == "animation") {
				String name;
				Animation value;
				if (LoadAnimation(element, name, value)) {
					properties.SetAnimation(name, value);
					return true;
				}
			}
			if (propertyType == "ref") {
				String name;
				Reference value;
				if (LoadReference(element, name, value)) {
					properties.SetReference(name, value);
					return true;
				}
			}
			return false;
		}
	private:
		void ImplLoadTransform(const tinyxml2::XMLElement* element, Transform& value)const
		{
			value.matrix = glm::identity<glm::mat4x4>();
			auto child = element->FirstChildElement();
			while (child) {
				std::string t_name = std::string(child->Value());
				if (t_name == "translate")
				{
					auto attrib_x = child->FindAttribute("x");
					auto attrib_y = child->FindAttribute("y");
					auto attrib_z = child->FindAttribute("z");
					float x = attrib_x ? attrib_x->FloatValue() : 0.0f;
					float y = attrib_y ? attrib_y->FloatValue() : 0.0f;
					float z = attrib_z ? attrib_z->FloatValue() : 0.0f;
					value.matrix = glm::translate(value.matrix, glm::vec3(x, y, z));
				}
				if (t_name == "rotate")
				{
					auto attrib_x = child->FindAttribute("x");
					auto attrib_y = child->FindAttribute("y");
					auto attrib_z = child->FindAttribute("z");
					auto attrib_angle = child->FindAttribute("angle");
					float x = attrib_x ? attrib_x->FloatValue() : 0.0f;
					float y = attrib_y ? attrib_y->FloatValue() : 0.0f;
					float z = attrib_z ? attrib_z->FloatValue() : 0.0f;
					float angle = attrib_angle ? attrib_angle->FloatValue() : 0.0f;
					value.matrix = glm::rotate(value.matrix, glm::radians(angle), glm::vec3(x, y, z));
				}
				if (t_name == "scale")
				{
					auto attrib_value = child->FindAttribute("value");
					if (attrib_value)
					{
						auto v = attrib_value->FloatValue();
						value.matrix = glm::scale(value.matrix, glm::vec3(v, v, v));
					}
					else {

						auto attrib_x = child->FindAttribute("x");
						auto attrib_y = child->FindAttribute("y");
						auto attrib_z = child->FindAttribute("z");
						float x = attrib_x ? attrib_x->FloatValue() : 0.0f;
						float y = attrib_y ? attrib_y->FloatValue() : 0.0f;
						float z = attrib_z ? attrib_z->FloatValue() : 0.0f;
						value.matrix = glm::scale(value.matrix, glm::vec3(x, y, z));
					}
				}
				if (t_name == "matrix")
				{
					std::vector<float> v;
					auto attrib_value = child->FindAttribute("value");
					std::string input = attrib_value->Value();
					auto first = input.begin();
					auto last = input.end();
					bool result = qi::phrase_parse(first, last, m_GrammarMatrix, qi::ascii::space, v);
					if (result && first == last)
					{
						value.matrix = glm::mat4x4(
							v[0], v[4], v[8] , v[12], 
							v[1], v[5], v[9] , v[13], 
							v[2], v[6], v[10], v[14], 
							v[3], v[7], v[11], v[15]
						) * value.matrix;
					}
				}
				if (t_name == "lookat")
				{
					auto attrib_origin = child->FindAttribute("origin");
					auto attrib_target = child->FindAttribute("target");
					auto attrib_up = child->FindAttribute("up");
					glm::vec3 origin;
					glm::vec3 target;
					glm::vec3 up;
					if (attrib_origin)
					{
						std::vector<float> v;
						std::string  input = attrib_origin->Value();
						auto first = input.begin();
						auto last = input.end();
						bool result = qi::phrase_parse(first, last, m_GrammarFloat3, qi::ascii::space, v);
						if (result && first == last)
						{
							origin = { v[0],v[1],v[2] };
						}
					}
					if (attrib_target)
					{
						std::vector<float> v;
						std::string  input = attrib_target->Value();
						auto first = input.begin();
						auto last = input.end();
						bool result = qi::phrase_parse(first, last, m_GrammarFloat3, qi::ascii::space, v);
						if (result && first == last)
						{
							target = { v[0],v[1],v[2] };
						}
					}
					if (attrib_up)
					{
						std::vector<float> v;
						std::string  input = attrib_up->Value();
						auto first = input.begin();
						auto last = input.end();
						bool result = qi::phrase_parse(first, last, m_GrammarFloat3, qi::ascii::space, v);
						if (result && first == last)
						{
							up = { v[0],v[1],v[2] };
						}
					}
					value.matrix = glm::lookAt(origin, target, up) * value.matrix;
				}
				child = child->NextSiblingElement();
			}
		}
	private:
		String               m_RootPath;
		GrammarFloat3Value   m_GrammarFloat3;
		GrammarMatrixValue   m_GrammarMatrix;
		GrammarSpectrumValue m_GrammarSpectrum;
	};
	struct SerializedData
	{
		struct {
			uint16_t       format        = 0;
			uint16_t       version       = 0;
			bool           hasNormal     = false;
			bool           hasTexCrd     = false;
			bool           hasVertColor  = false;
			bool           useFaceNormal = false;
			bool           useDoublePrec = true;
			std::string    shape_name    = "";
			std::uint64_t  numOfVert     = 0;
			std::uint64_t  numOfIndx     = 0;
		} header = {};
		struct Vertex32Data
		{
			using Float = float;
			std::vector<std::array<  Float, 3>>   positions  = {};
			std::vector<std::array<  Float, 3>>   normals    = {};
			std::vector<std::array<  Float, 2>>   texCoords  = {};
			std::vector<std::array<  Float, 3>>   vertColors = {};
		};
		struct Vertex64Data
		{
			using Float = double;
			std::vector<std::array<  Float, 3>>   positions  = {};
			std::vector<std::array<  Float, 3>>   normals    = {};
			std::vector<std::array<  Float, 2>>   texCoords  = {};
			std::vector<std::array<  Float, 3>>   vertColors = {};
		};
		using Index32Data = std::vector<std::array<uint32_t,3>>;
		using Index64Data = std::vector<std::array<uint64_t,3>>;
		std::variant<Vertex32Data, Vertex64Data> vertexData = Vertex32Data{};
		std::variant< Index32Data,  Index64Data>  indexData =  Index32Data{};
		bool Load(std::istream& istream)
		{
			uint16_t formatIdentifier = 0;
			istream.read((char*)&formatIdentifier, sizeof(uint16_t));
			if (formatIdentifier != 0x041c) {
				//Œ³‚É–ß‚·
				header.format = formatIdentifier;
				istream.seekg(-sizeof(uint16_t), std::ios::cur);
				return false;
			}
			else {
				uint16_t versionIdentifier = 0;
				istream.read((char*)&versionIdentifier, sizeof(uint16_t));
				header.version = versionIdentifier;
				boost::iostreams::filtering_istream in;
				in.push(boost::iostreams::zlib_decompressor());
				in.push(istream);
				uint32_t  mask = 0;
				in.read((char*)&mask, sizeof(uint32_t));
				header.hasNormal    = mask & 0x0001;
				header.hasTexCrd    = mask & 0x0002;
				header.hasVertColor = mask & 0x0008;
				header.useFaceNormal= mask & 0x0010;
				header.useDoublePrec= mask & 0x2000;
				std::vector<char> t_shape_name;
				char c;
				do {
					in.read(&c, sizeof(char));
					t_shape_name.push_back(c);
				} while (c != '\0');
				header.shape_name  = std::string(t_shape_name.data());
				uint64_t numOfVert = 0;
				in.read((char*)&numOfVert, sizeof(uint64_t));
				header.numOfVert   = numOfVert;
				uint64_t numOfIndx = 0;
				in.read((char*)&numOfIndx, sizeof(uint64_t));
				header.numOfIndx   = numOfIndx;
				if (!header.useDoublePrec) {
					Vertex32Data vertex_data = {};
					vertex_data.positions.resize(header.numOfVert);
					in.read((char*)vertex_data.positions.data(), sizeof(vertex_data.positions[0]) * vertex_data.positions.size());
					if (header.hasNormal) {
						vertex_data.normals.resize(header.numOfVert);
						in.read((char*)vertex_data.normals.data(), sizeof(vertex_data.normals[0]) * vertex_data.normals.size());
					}
					else {
						vertex_data.normals = {};
					}
					if (header.hasTexCrd) {
						vertex_data.texCoords.resize(header.numOfVert);
						in.read((char*)vertex_data.texCoords.data(), sizeof(vertex_data.texCoords[0]) * vertex_data.texCoords.size());
					}
					else {
						vertex_data.texCoords = {};
					}
					if (header.hasVertColor)
					{
						vertex_data.vertColors.resize(header.numOfVert);
						in.read((char*)vertex_data.vertColors.data(), sizeof(vertex_data.vertColors[0]) * vertex_data.vertColors.size());
					}
					else {
						vertex_data.vertColors = {};
					}
					vertexData = std::move(vertex_data);
				}
				else {
					Vertex64Data vertex_data = {};
					vertex_data.positions.resize(header.numOfVert);
					in.read((char*)vertex_data.positions.data(), sizeof(vertex_data.positions[0]) * vertex_data.positions.size());
					if (header.hasNormal) {
						vertex_data.normals.resize(header.numOfVert);
						in.read((char*)vertex_data.normals.data(), sizeof(vertex_data.normals[0]) * vertex_data.normals.size());
					}
					else {
						vertex_data.normals = {};
					}
					if (header.hasTexCrd) {
						vertex_data.texCoords.resize(header.numOfVert);
						in.read((char*)vertex_data.texCoords.data(), sizeof(vertex_data.texCoords[0]) * vertex_data.texCoords.size());
					}
					else {
						vertex_data.texCoords = {};
					}
					if (header.hasVertColor)
					{
						vertex_data.vertColors.resize(header.numOfVert);
						in.read((char*)vertex_data.vertColors.data(), sizeof(vertex_data.vertColors[0]) * vertex_data.vertColors.size());
					}
					else {
						vertex_data.vertColors = {};
					}
					vertexData = std::move(vertex_data);
				}
				if (header.numOfVert <= 0xFFFFFFFF)
				{
					Index32Data index_data = {};
					index_data.resize(header.numOfIndx);
					in.read((char*)index_data.data(), sizeof(index_data[0]) * index_data.size());
					indexData = std::move(index_data);
				}
				else {
					Index64Data index_data = {};
					index_data.resize(header.numOfIndx);
					in.read((char*)index_data.data(), sizeof(index_data[0]) * index_data.size());
					indexData = std::move(index_data);
				}
				return true;
			}
		}
	};
	struct SerializedFileData {
		struct {
			std::string             filename    = "";
			uint32_t                numMeshes   = 0;
			std::vector<uint64_t>   dataOffsets = {};
		} header;
		std::vector<SerializedData> data        = {};
		bool Load(const std::string& path) {
			std::ifstream file(path,std::ios::binary);
			if (!file.is_open()) {
				return false;
			}
			file.seekg(0, std::ios::beg);
			uint16_t formatIdentifier = 0;
			file.read((char*)&formatIdentifier, sizeof(uint16_t));
			if (formatIdentifier != 0x041c) {
				file.close();
				return false;
			}
			header.filename    = path;
			uint32_t numMeshes = 0;
			file.seekg(-sizeof(uint32_t), std::ios::end);
			file.read((char*)&numMeshes, sizeof(uint32_t));
			header.numMeshes   = numMeshes;
			header.dataOffsets.resize(numMeshes);
			file.seekg(-sizeof(uint32_t) - numMeshes * sizeof(uint64_t), std::ios::end);
			file.read((char*)header.dataOffsets.data(), sizeof(header.dataOffsets[0]) * header.dataOffsets.size());
			for (auto i = 0; i < header.dataOffsets.size(); ++i)
			{
				file.seekg(header.dataOffsets[i], std::ios::beg);
				mitsuba_loader::SerializedData s_data;
				if (s_data.Load(file)) {
					std::string content = "V";
					if (s_data.header.hasNormal) {
						content += "N";
					}
					if (s_data.header.hasTexCrd) {
						content += "T";
					}
					if (s_data.header.hasVertColor) {
						content += "C";
					}

					std::cout << "Success: " << "Token: " << std::setw(10) << std::dec << header.dataOffsets[i] << " Content: " << std::setw(5) << content << " Name: " << s_data.header.shape_name << "\n";
				}
				else {
					std::cout << "Failed : " << "Token: " << std::setw(10) << std::dec << header.dataOffsets[i] << " Name: " << std::setw(5) << s_data.header.shape_name << "\n";
				}
				data.emplace_back(std::move(s_data));
			}
			file.close();
			return true;
		}
	};
	struct Scene
	{
		uint32_t                                       versionMajor    = 0;
		uint32_t                                       versionMinor    = 0;
		uint32_t                                       versionPatch    = 0;
		std::unordered_map<String, ObjectPtr>          shapes          = {};
		std::unordered_map<String, ObjectPtr>          bsdfs           = {};
		std::unordered_map<String, ObjectPtr>          textures        = {};
		std::unordered_map<String, ObjectPtr>          subsurfaces     = {};
		std::unordered_map<String, ObjectPtr>          mediums         = {};
		std::unordered_map<String, ObjectPtr>          phases          = {};
		std::unordered_map<String, ObjectPtr>          volumes         = {};
		std::unordered_map<String, ObjectPtr>          emitters        = {};
		std::unordered_map<String, ObjectPtr>          sensors         = {};
		std::unordered_map<String, ObjectPtr>          integrators     = {};
		std::unordered_map<String, ObjectPtr>          rfilters        = {};
		std::unordered_map<String, SerializedFileData> serializedFiles = {};
	};
	void   LoadProperties(const tinyxml2::XMLElement* element, const mitsuba_loader::PropertyLoader& propertyLoader, mitsuba_loader::Properties& properties) {
		std::string objectTypeString = element->Name();
		auto elem1 = element->FirstChildElement();
		while (elem1) {
			std::string propertyType = elem1->Value();
			auto a_PropertyName = elem1->FindAttribute("name");
			if (a_PropertyName) {
				std::cout << objectTypeString << "." << a_PropertyName->Value() << "<" << propertyType << ">() =" << std::endl;
				//nested
				if (!propertyLoader.LoadProperties(elem1, properties)) {
					auto  objectType = mitsuba_loader::SupportType::Unknown;
					auto a_PluginName = elem1->FindAttribute("type");
					auto a_ObjectID = elem1->FindAttribute("id");
					std::string pluginName = a_PluginName ? a_PluginName->Value() : "";
					std::string objectID = a_ObjectID ? a_ObjectID->Value() : "";
					objectType = mitsuba_loader::ToSupportObjectType(propertyType);
					if (objectType != mitsuba_loader::SupportType::Unknown) {
						auto object = std::make_shared<mitsuba_loader::Object>(objectType, pluginName, objectID);
						LoadProperties(elem1, propertyLoader, object->GetProperties());
						properties.SetObjectPtr(a_PropertyName->Value(), object);
					}
				}
			}
			else {
				if (objectTypeString == "shape") {
					auto a_PluginName = elem1->FindAttribute("type");
					std::string pluginName = a_PluginName ? a_PluginName->Value() : "";
					if (propertyType == "bsdf")
					{
						auto object = std::make_shared<mitsuba_loader::Object>(mitsuba_loader::SupportType::Bsdf, pluginName, "");
						LoadProperties(elem1, propertyLoader, object->GetProperties());
						properties.SetObjectPtr("bsdf", object);
					}
				}
				if (objectTypeString == "sensor")
				{
					auto a_PluginName = elem1->FindAttribute("type");
					std::string pluginName = a_PluginName ? a_PluginName->Value() : "";
					auto a_ObjectID = elem1->FindAttribute("id");
					std::string objectID = a_ObjectID ? a_ObjectID->Value() : "";
					if (propertyType == "film")
					{
						auto object = std::make_shared<mitsuba_loader::Object>(mitsuba_loader::SupportType::Film, pluginName, "");
						LoadProperties(elem1, propertyLoader, object->GetProperties());
						properties.SetObjectPtr("film", object);
					}
					if (propertyType == "sampler")
					{
						auto object = std::make_shared<mitsuba_loader::Object>(mitsuba_loader::SupportType::Sampler, pluginName, "");
						LoadProperties(elem1, propertyLoader, object->GetProperties());
						properties.SetObjectPtr("sampler", object);
					}
					if (propertyType == "medium")
					{
						auto object = std::make_shared<mitsuba_loader::Object>(mitsuba_loader::SupportType::Medium, pluginName, objectID);
						LoadProperties(elem1, propertyLoader, object->GetProperties());
						properties.SetObjectPtr("medium", object);
					}
					if (propertyType == "ref") {
						auto a_ObjectID = elem1->FindAttribute("id");
						if (a_ObjectID) {
							properties.SetReference("medium", mitsuba_loader::Reference{ a_ObjectID->Value() });
						}
					}
				}
				if (objectTypeString == "film")
				{
					auto a_PluginName = elem1->FindAttribute("type");
					std::string pluginName = a_PluginName ? a_PluginName->Value() : "";
					if (propertyType == "rfilter")
					{
						auto object = std::make_shared<mitsuba_loader::Object>(mitsuba_loader::SupportType::Rfilter, pluginName, "");
						LoadProperties(elem1, propertyLoader, object->GetProperties());
						properties.SetObjectPtr("rfilter", object);
					}
				}
			}
			elem1 = elem1->NextSiblingElement();
		}
	};
	bool   LoadScene(const String& sceneXmlPath, Scene& scene) {
		std::string rootDir = std::filesystem::path(sceneXmlPath).parent_path().string();
		auto xmlDoc = tinyxml2::XMLDocument();
		if (xmlDoc.LoadFile(sceneXmlPath.c_str()) != tinyxml2::XML_SUCCESS)
		{
			return false;
		}
		auto propertyLoader = mitsuba_loader::PropertyLoader(rootDir);
		auto sceneElement   = xmlDoc.RootElement();
		{
			auto a_version = sceneElement->FindAttribute("version");
			if (!a_version) {
				return false;
			}
			{
				auto versionStr = std::string(a_version->Value());
				std::vector<int> versions;
				auto g_scene_version = GrammarSceneVersion();
				auto first = versionStr.begin();
				auto last  = versionStr.end();
				auto isSuccess = qi::phrase_parse(first, last, g_scene_version, qi::ascii::space, versions);
				if (isSuccess && first == last)
				{
					scene.versionMajor = versions[0];
					scene.versionMinor = versions[1];
					scene.versionPatch = versions[2];
				}
			}
			std::cout << scene.versionMajor << "." << scene.versionMinor << "." << scene.versionPatch << std::endl;
			auto elem = sceneElement->FirstChildElement();
			while (elem) {
				auto a_PluginName = elem->FindAttribute("type");
				auto a_ObjectID = elem->FindAttribute("id");
				std::string objectTypeString = elem->Name();
				auto  objectType = mitsuba_loader::SupportType::Unknown;
				std::string pluginName = a_PluginName ? a_PluginName->Value() : "";
				std::string objectID = a_ObjectID ? a_ObjectID->Value() : "";
				if (objectTypeString == "shape") {
					objectType = mitsuba_loader::SupportType::Shape;
					std::cout << objectTypeString << std::endl;
					auto object = std::make_shared<mitsuba_loader::Object>(objectType, pluginName, objectID);
					scene.shapes[object->GetObjectID()] = object;
					LoadProperties(elem, propertyLoader, object->GetProperties());
				}
				if (objectTypeString == "bsdf") {
					objectType = mitsuba_loader::SupportType::Bsdf;
					std::cout << objectTypeString << std::endl;
					auto object = std::make_shared<mitsuba_loader::Object>(objectType, pluginName, objectID);
					scene.bsdfs[object->GetObjectID()] = object;
					LoadProperties(elem, propertyLoader, object->GetProperties());
				}
				if (objectTypeString == "texture") {
					objectType = mitsuba_loader::SupportType::Texture;
					std::cout << objectTypeString << std::endl;
					auto object = std::make_shared<mitsuba_loader::Object>(objectType, pluginName, objectID);
					scene.textures[object->GetObjectID()] = object;
					LoadProperties(elem, propertyLoader, object->GetProperties());
				}
				if (objectTypeString == "subsurface") {
					objectType = mitsuba_loader::SupportType::Subsurface;
					std::cout << objectTypeString << std::endl;
					auto object = std::make_shared<mitsuba_loader::Object>(objectType, pluginName, objectID);
					scene.subsurfaces[object->GetObjectID()] = object;
					LoadProperties(elem, propertyLoader, object->GetProperties());
				}
				if (objectTypeString == "medium") {
					objectType = mitsuba_loader::SupportType::Medium;
					std::cout << objectTypeString << std::endl;
					auto object = std::make_shared<mitsuba_loader::Object>(objectType, pluginName, objectID);
					scene.mediums[object->GetObjectID()] = object;
					LoadProperties(elem, propertyLoader, object->GetProperties());
				}
				if (objectTypeString == "phase") {
					objectType = mitsuba_loader::SupportType::Phase;
					std::cout << objectTypeString << std::endl;
					auto object = std::make_shared<mitsuba_loader::Object>(objectType, pluginName, objectID);
					scene.phases[object->GetObjectID()] = object;
					LoadProperties(elem, propertyLoader, object->GetProperties());
				}
				if (objectTypeString == "volume") {
					objectType = mitsuba_loader::SupportType::Volume;
					std::cout << objectTypeString << std::endl;
					auto object = std::make_shared<mitsuba_loader::Object>(objectType, pluginName, objectID);
					scene.volumes[object->GetObjectID()] = object;
					LoadProperties(elem, propertyLoader, object->GetProperties());
				}
				if (objectTypeString == "emitter") {
					objectType = mitsuba_loader::SupportType::Emitter;
					std::cout << objectTypeString << std::endl;
					auto object = std::make_shared<mitsuba_loader::Object>(objectType, pluginName, objectID);
					scene.emitters[object->GetObjectID()] = object;
					LoadProperties(elem, propertyLoader, object->GetProperties());
				}
				if (objectTypeString == "sensor") {
					objectType = mitsuba_loader::SupportType::Sensor;
					std::cout << objectTypeString << std::endl;
					auto object = std::make_shared<mitsuba_loader::Object>(objectType, pluginName, objectID);
					scene.sensors[object->GetObjectID()] = object;
					LoadProperties(elem, propertyLoader, object->GetProperties());
				}
				if (objectTypeString == "integrator") {
					objectType = mitsuba_loader::SupportType::Integrator;
					std::cout << objectTypeString << std::endl;
					auto object = std::make_shared<mitsuba_loader::Object>(objectType, pluginName, objectID);
					scene.integrators[object->GetObjectID()] = object;
					LoadProperties(elem, propertyLoader, object->GetProperties());
				}
				elem = elem->NextSiblingElement();
			}
			for (auto& [name, shape] : scene.shapes)
			{
				if (shape->GetPluginName() == "serialized") {
					auto filename = shape->GetProperties().GetString("filename");
					if (scene.serializedFiles.count(filename) == 0) {
						scene.serializedFiles[filename] = {};
						if (!scene.serializedFiles[filename].Load(propertyLoader.GetRootPath() + "/" + filename)) {
							std::cout << "Failed To Load " << filename << "\n";
						}
					}
				}
			}
		}
		return true;
	}
	void   WriteAsObjLike(const SerializedData& serializedData, const std::string& objectName, const std::string& materialName, std::ostream& os, const Transform& trasnform = Transform{glm::identity<glm::mat4x4>()}, size_t indexOffset = 0)
	{
		os << "o " << objectName << "\n";
		auto WriteVertexData = [&os, trasnform](const auto& header,const auto& vertexData)->void {
			for (auto i = 0; i < header.numOfVert; ++i) {
				auto vertex4 = glm::vec4(vertexData.positions[i][0], vertexData.positions[i][1], vertexData.positions[i][2], 1.0f);
				vertex4 = trasnform.matrix * vertex4;
				os << "v " << vertex4.x << " " << vertex4.y << " " << vertex4.z << "\n";
			}
			if (header.hasNormal) {
				for (auto i = 0; i < header.numOfVert; ++i) {
					auto normal3 = glm::vec3(vertexData.normals[i][0], vertexData.normals[i][1], vertexData.normals[i][2]);
					normal3 = glm::inverseTranspose(glm::mat3x3(trasnform.matrix)) * normal3;
					os << "vn " << normal3.x << " " << normal3.y << " " << normal3.z << "\n";
				}
				
			}
			if (header.hasTexCrd) {
				for (auto i = 0; i < header.numOfVert; ++i) {
					os << "vt " << vertexData.texCoords[i][0] << " " << vertexData.texCoords[i][1]<< "\n";
				}
			}
		};
		auto WriteIndexData = [&os](const auto& header, const auto& indexData)->void {
			if (header.hasTexCrd && header.hasNormal) {
				for (auto i = 0; i < header.numOfIndx; ++i) {
					os << "f " << 1 + indexData[i][0] << "/" << 1 + indexData[i][0] << "/" << 1 + indexData[i][0] << " ";
					os << 1 + indexData[i][1] << "/" << 1 + indexData[i][1] << "/" << 1 + indexData[i][1] << " ";
					os << 1 + indexData[i][2] << "/" << 1 + indexData[i][2] << "/" << 1 + indexData[i][2] << "\n";
				}
			}
			else if (header.hasTexCrd) {
				for (auto i = 0; i < header.numOfIndx; ++i) {
					os << "f " << 1 + indexData[i][0] << "/" << 1 + indexData[i][0] <<" ";
					os << 1 + indexData[i][1] << "/" << 1 + indexData[i][1] << " ";
					os << 1 + indexData[i][2] << "/" << 1 + indexData[i][2] << "\n";
				}
			}
			else if (header.hasNormal) {
				for (auto i = 0; i < header.numOfIndx; ++i) {
					os << "f " << 1 + indexData[i][0] << "//" << 1 + indexData[i][0] << " ";
					os << 1 + indexData[i][1] << "//" << 1 + indexData[i][1] << " ";
					os << 1 + indexData[i][2] << "//" << 1 + indexData[i][2] << "\n";
				}
			}
			else {
				for (auto i = 0; i < header.numOfIndx; ++i) {
					os << "f " << 1 + indexData[i][0] << " ";
					os << 1 + indexData[i][1] << " ";
					os << 1 + indexData[i][2] << "\n";
				}
			}
		};
		if (serializedData.header.useDoublePrec) {
			auto& vertexData = std::get<mitsuba_loader::SerializedData::Vertex64Data>(serializedData.vertexData);
			WriteVertexData(serializedData.header, vertexData);
		}
		else {
			auto& vertexData = std::get<mitsuba_loader::SerializedData::Vertex32Data>(serializedData.vertexData);
			WriteVertexData(serializedData.header, vertexData);
		}
		if (serializedData.header.useFaceNormal) {
			os << "s on\n";
		}
		else {
			os << "s off\n";
		}
		os << "usemtl " << materialName << "\n";
		if (serializedData.header.numOfVert > 0xFFFFFFFF) {
			auto& indexData = std::get<mitsuba_loader::SerializedData::Index64Data>(serializedData.indexData);
			WriteIndexData(serializedData.header, indexData);
		}
		else {
			auto& indexData = std::get<mitsuba_loader::SerializedData::Index32Data>(serializedData.indexData);
			WriteIndexData(serializedData.header, indexData);
		}
	}
}
int main() {
	//Basic
	auto scene = mitsuba_loader::Scene{};
	if (!mitsuba_loader::LoadScene(TEST_TEST_XML_DATA_PATH"/Scenes/Pool/pool.xml", scene)) {
		return -1;
	}

	for (auto& [name, serializedData] : scene.serializedFiles) {
		auto current_serialized_dir = std::filesystem::path(std::string(TEST_TEST_XML_DATA_PATH) + "/Models/Pool/" + name + "/");
		std::filesystem::create_directory(current_serialized_dir);
	}
	for (auto& [name,shape] : scene.shapes) {
		if (shape->GetPluginName() == "serialized") {
			std::cout << "    Serialized: " << shape->GetProperties().GetString("filename") << "." << shape->GetProperties().GetInteger("shapeIndex") << "\n";
			auto& serializedFileData    = scene.serializedFiles  [shape->GetProperties().GetString("filename")   ];
			auto& serializedData        = serializedFileData.data[shape->GetProperties().GetInteger("shapeIndex")];
			if (shape->GetProperties().Has("bsdf")) {
				auto objectType = shape->GetProperties().GetType("bsdf");
				auto transform  = mitsuba_loader::Transform{ glm::identity < glm::mat4x4>() };
				if (shape->GetProperties().Has("toWorld")) {
					transform = shape->GetProperties().GetTransform("toWorld");
					std::cout << "    Serialized: " << shape->GetProperties().GetString("filename") << "." << shape->GetProperties().GetInteger("shapeIndex") << " : " <<  glm::to_string(transform.matrix) << std::endl;
				}
				std::string materialName = "";
				if (objectType == mitsuba_loader::SupportType::Reference) {
					materialName = shape->GetProperties().GetReference("bsdf").objectID;
					auto bsdf    = scene.bsdfs[materialName];
					scene.bsdfs.erase(materialName);
					boost::algorithm::replace_all(materialName, "#", "-");
					shape->GetProperties().SetReference("bsdf", { materialName });
					bsdf->SetObjectID({ materialName });
					scene.bsdfs[materialName] = bsdf;
				}
				if (objectType == mitsuba_loader::SupportType::Bsdf) {
					auto object  = shape->GetProperties().GetObjectPtr("bsdf");
					materialName = object->GetObjectID();
					boost::algorithm::replace_all(materialName, "#", "-");
					object->SetObjectID(materialName);
				}
				
				std::ofstream file(std::string(TEST_TEST_XML_DATA_PATH) + "/Models/Pool/" + shape->GetProperties().GetString("filename") +"/" + name + ".obj", std::ios::binary);
				file << "mtllib ../pool.mtl\n";
				mitsuba_loader::WriteAsObjLike(serializedData, serializedData.header.shape_name, materialName, file, transform,0);
				file.close();
			}
		}
		else {
			std::cout << "Not Serialized\n";
		}
	}
	{
		std::ofstream file(TEST_TEST_XML_DATA_PATH"/Models/Pool/pool.mtl", std::ios::binary);
		for (auto& [name, bsdf] : scene.bsdfs)
		{
			file << "newmtl " << name << "\n";
			if (bsdf->GetPluginName() == "diffuse")
			{
				int   illum    = 2;
				float Ns       = 10.0f;
				float Ni       = 1.0f;
				float d        = 1.0f;
				auto  Tf	   = std::array<float, 3>{ 1.0f, 1.0f, 1.0f };
				auto  Ka       = std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
				auto  Kd       = std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
				auto  Ke       = std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
				auto  Ks       = std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
				auto  map_Kd   = std::string("");
				if (bsdf->GetProperties().Has("reflectance")) {
					if (bsdf->GetProperties().GetType("reflectance") == mitsuba_loader::SupportType::RGBColor)
					{
						Kd = bsdf->GetProperties().GetRGBColor("reflectance").GetFloat3();
					}
					if (bsdf->GetProperties().GetType("reflectance") == mitsuba_loader::SupportType::Spectrum)
					{
						Kd = std::array<float, 3>{
							bsdf->GetProperties().GetSpectrum("reflectance").GetWaveWeights()[0],
							bsdf->GetProperties().GetSpectrum("reflectance").GetWaveWeights()[0],
							bsdf->GetProperties().GetSpectrum("reflectance").GetWaveWeights()[0]
						};
					}
					if (bsdf->GetProperties().GetType("reflectance") == mitsuba_loader::SupportType::Texture) {
						Kd = std::array<float, 3>{
							1.0f, 1.0f, 1.0f
						};
						auto object = bsdf->GetProperties().GetObjectPtr("reflectance");
						if (object->GetPluginName() == "bitmap")
						{
							map_Kd = object->GetProperties().GetString("filename");
						}
					}
					if (bsdf->GetProperties().GetType("reflectance") == mitsuba_loader::SupportType::Reference)
					{
						Kd = std::array<float, 3>{
							1.0f, 1.0f, 1.0f
						};
						auto object = scene.textures[bsdf->GetProperties().GetReference("reflectance").objectID];
						if (object->GetPluginName() == "bitmap")
						{
							map_Kd = object->GetProperties().GetString("filename");
						}
					}
				}
				file << "	Ns " << Ns << "\n";
				file << "	Ni " << Ni << "\n";
				file << "	d "  << d  << "\n";
				file << "	Tr 0.0000\n";
				file << "	Tf " << Tf[0] << " " << Tf[1] << " " << Tf[2] << "\n";
				file << "	illum " << illum << "\n";
				file << "	Ka " << Ka[0] << " " << Ka[1] << " " << Ka[2] << "\n";
				file << "	Kd " << Kd[0] << " " << Kd[1] << " " << Kd[2] << "\n";
				file << "	Ks " << Ks[0] << " " << Ks[1] << " " << Ks[2] << "\n";
				file << "	Ke " << Ke[0] << " " << Ke[1] << " " << Ke[2] << "\n";
				if (!map_Kd.empty()) {
					file << "	map_Ka ../" << map_Kd << "\n";
					file << "	map_Kd ../" << map_Kd << "\n";
				}
				file << "\n";
			}
			if (bsdf->GetPluginName() == "phong")
			{
				int   illum    = 2;
				float Ns       = 0.0f;
				float Ni       = 1.0f;
				float d        = 1.0f;
				auto  Tf	   = std::array<float, 3>{ 1.0f, 1.0f, 1.0f };
				auto  Ka	   = std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
				auto  Kd	   = std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
				auto  Ke	   = std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
				auto  Ks	   = std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
				auto  map_Kd   = std::string("");
				auto  map_Ks   = std::string("");
				auto  map_Ns   = std::string("");
				if (bsdf->GetProperties().Has( "diffuseReflectance")) {
					if (bsdf->GetProperties().GetType("diffuseReflectance") == mitsuba_loader::SupportType::RGBColor)
					{
						Kd = bsdf->GetProperties().GetRGBColor("diffuseReflectance").GetFloat3();
					}
					if (bsdf->GetProperties().GetType("diffuseReflectance") == mitsuba_loader::SupportType::Spectrum)
					{
						Kd = std::array<float, 3>{
							bsdf->GetProperties().GetSpectrum("diffuseReflectance").GetWaveWeights()[0],
							bsdf->GetProperties().GetSpectrum("diffuseReflectance").GetWaveWeights()[0],
							bsdf->GetProperties().GetSpectrum("diffuseReflectance").GetWaveWeights()[0]
						};
					}
					if (bsdf->GetProperties().GetType("diffuseReflectance") == mitsuba_loader::SupportType::Texture) {
						Kd = std::array<float, 3>{
							1.0f, 1.0f, 1.0f
						};
						auto object = bsdf->GetProperties().GetObjectPtr("diffuseReflectance");
						if (object->GetPluginName() == "bitmap")
						{
							map_Kd = object->GetProperties().GetString("filename");
						}
					}
					if (bsdf->GetProperties().GetType("diffuseReflectance") == mitsuba_loader::SupportType::Reference)
					{
						Kd = std::array<float, 3>{
							1.0f, 1.0f, 1.0f
						};
						auto object = scene.textures[bsdf->GetProperties().GetReference("diffuseReflectance").objectID];
						if (object->GetPluginName() == "bitmap")
						{
							map_Kd = object->GetProperties().GetString("filename");
						}
					}
				}
				if (bsdf->GetProperties().Has("specularReflectance")) {
					if (bsdf->GetProperties().GetType("specularReflectance") == mitsuba_loader::SupportType::RGBColor )
					{
						Ks = bsdf->GetProperties().GetRGBColor("specularReflectance").GetFloat3();
					}
					if (bsdf->GetProperties().GetType("specularReflectance") == mitsuba_loader::SupportType::Spectrum )
					{
						Ks = std::array<float, 3>{
							bsdf->GetProperties().GetSpectrum("specularReflectance").GetWaveWeights()[0],
								bsdf->GetProperties().GetSpectrum("specularReflectance").GetWaveWeights()[0],
								bsdf->GetProperties().GetSpectrum("specularReflectance").GetWaveWeights()[0]
						};
					}
					if (bsdf->GetProperties().GetType("specularReflectance") == mitsuba_loader::SupportType::Texture  ) {
						Ks = std::array<float, 3>{
							1.0f, 1.0f, 1.0f
						};
						auto object = bsdf->GetProperties().GetObjectPtr("specularReflectance");
						if (object->GetPluginName() == "bitmap")
						{
							map_Ks = object->GetProperties().GetString("filename");
						}
					}
					if (bsdf->GetProperties().GetType("specularReflectance") == mitsuba_loader::SupportType::Reference)
					{
						Ks = std::array<float, 3>{
							1.0f, 1.0f, 1.0f
						};
						auto object = scene.textures[bsdf->GetProperties().GetReference("specularReflectance").objectID];
						if (object->GetPluginName() == "bitmap")
						{
							map_Ks = object->GetProperties().GetString("filename");
						}
					}
				}
				if (bsdf->GetProperties().Has("exponent")) {
					if (bsdf->GetProperties().GetType("exponent") == mitsuba_loader::SupportType::Float)
					{
						Ns = bsdf->GetProperties().GetFloat("exponent");
					}
					if (bsdf->GetProperties().GetType("exponent") == mitsuba_loader::SupportType::Texture)
					{
						Ns = 1.0f;
						auto object = bsdf->GetProperties().GetObjectPtr("exponent");
						if (object->GetPluginName() == "bitmap")
						{
							map_Ks  = object->GetProperties().GetString("filename");
						}
					}
					if (bsdf->GetProperties().GetType("exponent") == mitsuba_loader::SupportType::Reference)
					{
						Ns = 1.0f;
						auto reference = bsdf->GetProperties().GetReference("exponent");
						auto object    = scene.textures[reference.objectID];
						if (object->GetPluginName() == "bitmap")
						{
							map_Ks = object->GetProperties().GetString("filename");
						}
					}
				}
				file << "	Ns " << Ns << "\n";
				file << "	Ni " << Ni << "\n";
				file << "	d " << d << "\n";
				file << "	Tr 0.0000\n";
				file << "	Tf " << Tf[0] << " " << Tf[1] << " " << Tf[2] << "\n";
				file << "	illum " << illum << "\n";
				file << "	Ka " << Ka[0] << " " << Ka[1] << " " << Ka[2] << "\n";
				file << "	Kd " << Kd[0] << " " << Kd[1] << " " << Kd[2] << "\n";
				file << "	Ks " << Ks[0] << " " << Ks[1] << " " << Ks[2] << "\n";
				file << "	Ke " << Ke[0] << " " << Ke[1] << " " << Ke[2] << "\n";
				if (!map_Kd.empty()) {
					file << "	map_Ka ../" << map_Kd << "\n";
					file << "	map_Kd ../" << map_Kd << "\n";
				}
				if (!map_Ks.empty()) {
					file << "	map_Ks ../" << map_Ks << "\n";
				}
				if (!map_Ns.empty()) {
					file << "	map_Ns ../" << map_Ns << "\n";
				}
				file << "\n";
			}
			if (bsdf->GetPluginName() == "dielectric")
			{
				float intIOR   = bsdf->GetProperties().GetFloat("intIOR");
				float extIOR   = bsdf->GetProperties().GetFloat("extIOR");
				float shinness = 10.0f;
				int   illum    = 7;
				float Ns       = 100.0f;
				float Ni       = intIOR / extIOR;
				float d        = 1.0f;
				auto  Tf       = std::array<float, 3>{ 1.0f ,1.0f ,1.0f };
				auto  Ka       = std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
				auto  Kd       = std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
				auto  Ke       = std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
				auto  Ks       = std::array<float, 3>{ 1.0f, 1.0f, 1.0f };
				auto  map_Ks   = std::string("");
				if (bsdf->GetProperties().Has("specularReflectance"  )) {
					if (bsdf->GetProperties().GetType("specularReflectance") == mitsuba_loader::SupportType::RGBColor)
					{
						Ks = bsdf->GetProperties().GetRGBColor("specularReflectance").GetFloat3();
					}
					if (bsdf->GetProperties().GetType("specularReflectance") == mitsuba_loader::SupportType::Spectrum)
					{
						Ks = std::array<float, 3>{
							bsdf->GetProperties().GetSpectrum("specularReflectance").GetWaveWeights()[0],
							bsdf->GetProperties().GetSpectrum("specularReflectance").GetWaveWeights()[0],
							bsdf->GetProperties().GetSpectrum("specularReflectance").GetWaveWeights()[0]
						};
					}
					if (bsdf->GetProperties().GetType("specularReflectance") == mitsuba_loader::SupportType::Texture ) {
						Ks = std::array<float, 3>{
							1.0f,1.0f,1.0f
						};
						auto object = bsdf->GetProperties().GetObjectPtr("specularReflectance");
						if (object->GetPluginName() == "bitmap")
						{
							map_Ks = object->GetProperties().GetString("filename");
						}
					}
					if (bsdf->GetProperties().GetType("specularReflectance") == mitsuba_loader::SupportType::Reference)
					{
						Ks = std::array<float, 3>{
							1.0f, 1.0f, 1.0f
						};
						auto object = scene.textures[bsdf->GetProperties().GetReference("specularReflectance").objectID];
						if (object->GetPluginName() == "bitmap")
						{
							map_Ks = object->GetProperties().GetString("filename");
						}
					}

				}
				if (bsdf->GetProperties().Has("specularTransmittance"))
				{
					if (bsdf->GetProperties().GetType("specularTransmittance")==mitsuba_loader::SupportType::RGBColor)
					{
						Tf = bsdf->GetProperties().GetRGBColor("specularTransmittance").GetFloat3();
					}
					if (bsdf->GetProperties().GetType("specularTransmittance")==mitsuba_loader::SupportType::Spectrum)
					{
						Tf = std::array<float, 3>{
							bsdf->GetProperties().GetSpectrum("specularTransmittance").GetWaveWeights()[0],
							bsdf->GetProperties().GetSpectrum("specularTransmittance").GetWaveWeights()[0],
							bsdf->GetProperties().GetSpectrum("specularTransmittance").GetWaveWeights()[0]
						};
					}

				}
				file << "	Ns " << Ns << "\n";
				file << "	Ni " << Ni << "\n";
				file << "	d " << d << "\n";
				file << "	Tr 0.0000\n";
				file << "	Tf " << Tf[0] << " " << Tf[1] << " " << Tf[2] << "\n";
				file << "	illum " << illum << "\n";
				file << "	Ka " << Ka[0] << " " << Ka[1] << " " << Ka[2] << "\n";
				file << "	Kd " << Kd[0] << " " << Kd[1] << " " << Kd[2] << "\n";
				file << "	Ks " << Ks[0] << " " << Ks[1] << " " << Ks[2] << "\n";
				file << "	Ke " << Ke[0] << " " << Ke[1] << " " << Ke[2] << "\n";
				if (!map_Ks.empty()) {
					file << "	map_Ks ../" << map_Ks << "\n";
				}
				file << "\n";

			}
		}
		file.close();
	}
	return 0;
}